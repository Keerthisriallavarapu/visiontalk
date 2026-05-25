"""Vision-language model wrapper.

We use a pre-trained VLM (LLaVA-NeXT or Llama 3.2 Vision by default) and
fine-tune via LoRA. The wrapper:
- Loads the base model and processor.
- Handles image preprocessing.
- Generates text given (image, prompt).
- Streams output token by token when requested.

LoRA configs target the LLM backbone's attention projections. The vision
encoder stays frozen — fine-tuning a CLIP-style encoder on small data
overfits hard and degrades general perception.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    dtype: str = "bfloat16"
    device: str = "auto"
    quantization: str | None = None  # "int8" | "int4" | None
    use_flash_attention: bool = True


class VLMWrapper:
    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig()
        self._load()

    def _load(self):
        from transformers import (
            AutoProcessor,
            LlavaNextForConditionalGeneration,
        )

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.dtype]

        kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device,
        }
        if self.config.use_flash_attention:
            kwargs["attn_implementation"] = "sdpa"  # fallback if flash-attn missing

        if self.config.quantization == "int4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization == "int8":
            kwargs["load_in_8bit"] = True

        log.info("Loading %s ...", self.config.model_id)
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config.model_id, **kwargs
        )
        self.model.eval()
        log.info("Loaded model. Device: %s", self.model.device)

    def attach_lora(self, lora_path: str | Path) -> None:
        """Apply a LoRA adapter (from training/) to the loaded base model."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, str(lora_path))
        log.info("Attached LoRA from %s", lora_path)

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        do_sample: bool = False,
    ) -> str:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Build the chat template — exact format varies by model. LLaVA-NeXT
        # uses [INST] ... [/INST]; processors handle this when given a chat list.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image, text=prompt_text, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        # Strip the prompt portion
        gen = output[0, inputs.input_ids.shape[1]:]
        text = self.processor.decode(gen, skip_special_tokens=True)
        return text.strip()

    def generate_stream(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Yield generated tokens as they arrive. Used by the SSE endpoint.

        Implementation note: TextIteratorStreamer from transformers runs
        generation in a background thread and yields decoded chunks.
        """
        from threading import Thread

        from transformers import TextIteratorStreamer

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image, text=prompt_text, return_tensors="pt"
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            streamer=streamer,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for chunk in streamer:
                yield chunk
        finally:
            thread.join()

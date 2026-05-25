"""LoRA fine-tuning for vision-language models.

Why LoRA: full fine-tuning of a 7-13B VLM needs >80GB VRAM. LoRA reduces
trainable parameters to ~0.5% of the model, fitting in 24GB with the vision
encoder frozen.

Targeting: we attach LoRA only to the LLM backbone's q_proj and v_proj.
Some setups also LoRA the vision encoder; we don't because (a) it overfits
on small VQA datasets and (b) the gain doesn't justify the param count.

Dataset format: a list of {image_path, question, answer} dicts. We use the
LLaVA chat format for the prompt.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 16              # rank
    alpha: int = 32          # LoRA alpha (scaling = alpha/r)
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 4
    grad_accumulation_steps: int = 4
    n_epochs: int = 1
    warmup_steps: int = 100
    max_seq_length: int = 1024
    save_every_n_steps: int = 500
    seed: int = 42


class VQADataset(Dataset):
    """A list of (image_path, question, answer) examples."""

    def __init__(self, items: list[dict], processor, max_length: int = 1024):
        self.items = items
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        question = item["question"]
        answer = item["answer"]

        conversation = [
            {"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": answer}]},
        ]
        # apply_chat_template with add_generation_prompt=False emits the full
        # conversation; the answer becomes a target for the LM loss.
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )
        encoded = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Squeeze batch dim
        return {k: v.squeeze(0) for k, v in encoded.items()}


def load_dataset_jsonl(path: str | Path) -> list[dict]:
    """Each line: {"image_path": ..., "question": ..., "answer": ...}"""
    items = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def train_lora(
    base_model_id: str,
    train_data: list[dict],
    out_dir: str | Path,
    lora_config: LoRAConfig | None = None,
    train_config: TrainConfig | None = None,
    val_data: list[dict] | None = None,
) -> str:
    """Run LoRA fine-tuning. Returns the path to the saved adapter."""
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoProcessor,
        LlavaNextForConditionalGeneration,
        get_linear_schedule_with_warmup,
    )

    lora_config = lora_config or LoRAConfig()
    train_config = train_config or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(train_config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading base model %s ...", base_model_id)

    processor = AutoProcessor.from_pretrained(base_model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Freeze vision encoder
    for p in model.vision_tower.parameters():
        p.requires_grad = False

    peft_cfg = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
        bias=lora_config.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    dataset = VQADataset(train_data, processor, max_length=train_config.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    total_steps = (len(loader) // train_config.grad_accumulation_steps) * train_config.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    optimizer.zero_grad()
    for epoch in range(train_config.n_epochs):
        log.info("=== Epoch %d/%d ===", epoch + 1, train_config.n_epochs)
        running_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss / train_config.grad_accumulation_steps
            loss.backward()
            running_loss += loss.item() * train_config.grad_accumulation_steps

            if (step + 1) % train_config.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    log.info("step=%d loss=%.4f lr=%.2e",
                             global_step, running_loss / 50, scheduler.get_last_lr()[0])
                    running_loss = 0.0

                if global_step % train_config.save_every_n_steps == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    log.info("Saved checkpoint: %s", ckpt)

    final_dir = out_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    log.info("Saved final LoRA to %s", final_dir)
    return str(final_dir)

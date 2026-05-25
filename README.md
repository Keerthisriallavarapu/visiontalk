# VisionTalk

A vision-language assistant: upload an image, ask a question, get a streamed answer. Built around a pre-trained VLM (LLaVA-NeXT by default) with LoRA fine-tuning, INT4 quantization, and a streaming HTTP API.

## What's here

- **VLM wrapper** over LLaVA-NeXT (swap to any HF VLM by changing the config).
- **LoRA fine-tuning** pipeline for VQA-style datasets — fits in 24GB VRAM with the vision encoder frozen.
- **INT4/INT8 quantization** via bitsandbytes (optional; needs the `quantize` extra).
- **Streaming HTTP API** with SSE for incremental token output.
- **VQAv2-style evaluation** with proper normalization and the min(matches/3, 1.0) grader.
- **Next.js frontend** with drag-and-drop upload and live streaming.

## Quick start

```bash
# Install
pip install -e ".[quantize,dev]"
huggingface-cli login

# One-shot query (downloads ~14GB on first run; quantized fits in 8GB)
visiontalk ask path/to/image.jpg "What is the person holding?"

# Train LoRA on your dataset
visiontalk train data/train.jsonl --epochs 1 --out-dir artifacts/lora

# Evaluate on a held-out set
visiontalk eval data/val.jsonl --lora artifacts/lora/final

# Serve
visiontalk serve --quantization int4 --lora artifacts/lora/final &
cd frontend && pnpm install && pnpm dev
```

Dataset JSONL format:
```jsonl
{"image_path": "data/img/1.jpg", "question": "What color is the car?", "answer": "red", "answers": ["red", "red", "crimson"]}
```

## Why LoRA + frozen vision encoder

Full fine-tuning a 7-13B VLM exceeds 80GB VRAM. LoRA reduces trainable params to ~0.5% of the model. The vision encoder stays frozen because:
- Fine-tuning CLIP-style encoders on small VQA datasets overfits aggressively.
- The win is small (perceptual features generalize well) but the cost is large.
- The bottleneck is usually the LLM backbone's reasoning, not the vision features.

This matches what most production VLM teams do — they LoRA the LLM, leave the vision tower alone, and re-train the projector for major domain shifts.

## Performance targets

On a single RTX 4090 with INT4 quantization:

| Metric | Target |
|---|---|
| Time-to-first-token (TTFT) | ~600ms |
| Tokens/sec (decode) | ~25 |
| Peak VRAM | ~8GB |
| VQAv2 val accuracy (subset, base model) | ~0.72 |

Numbers will move with model choice and prompt. Reproduce with `visiontalk eval` and `visiontalk ask` after `python scripts/profile_inference.py`.

## Project layout

```
visiontalk/
├── visiontalk/
│   ├── models/      # VLM wrapper + chat template plumbing
│   ├── training/    # LoRA training loop
│   ├── serving/     # FastAPI with SSE streaming
│   ├── eval/        # VQAv2 graders
│   └── cli.py
├── frontend/        # Next.js drag-drop UI
├── scripts/
└── tests/
```

## What's not here yet

- **Region grounding**: drawing bounding boxes from the model's output. LLaVA-NeXT doesn't output coordinates natively; you'd plug in something like GroundingDINO for the box-prediction layer.
- **Video**: needs a different model family (Video-LLaVA, LLaVA-Video).
- **vLLM backend**: faster than transformers' `generate()`. Would replace the model wrapper with a vLLM client.

## License

Apache 2.0.

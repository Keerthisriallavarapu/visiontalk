# Engineering Decisions

## D-001: LLaVA-NeXT as the default base, with the option to swap

**Status:** Accepted

**Context.** Many VLM choices: BLIP-2, MiniGPT-4, LLaVA-1.5, LLaVA-NeXT, InstructBLIP, Qwen-VL, Llama 3.2 Vision, etc.

**Decision.** Default to LLaVA-NeXT 7B. Make the model id a config knob.

**Why.**
- LLaVA-NeXT has the cleanest HuggingFace integration; processor handles multi-image and chat formatting.
- 7B fits in 24GB easily and INT4 fits in 8GB.
- Quality is competitive with closed-source for VQA-style tasks.

**Things that don't matter as much as I expected.** Vision encoder choice (CLIP vs SigLIP) is a minor influence on VQA accuracy compared to the LLM backbone. Spend optimization effort on the LLM half.

---

## D-002: Freeze the vision encoder; LoRA the LLM

**Status:** Accepted

See README. Short version: vision encoder fine-tuning overfits on small data, the win is small even on large data, and the cost (memory, time) is high.

---

## D-003: SSE for streaming, not WebSockets

**Status:** Accepted

**Context.** Streaming tokens from generation to the frontend.

**Decision.** SSE.

**Why.**
- SSE is HTTP, plays nicely with proxies, no library on the frontend (built-in `EventSource`).
- Single direction (server -> client) is exactly what we need.
- WebSockets add bidirectional capability we don't use.
- Multipart upload + SSE response is a slightly awkward combo (one endpoint can't do both), so we use `/ask/stream` for SSE and the same multipart upload contract.

**When you'd switch.** Two-way features like interrupting generation mid-stream. SSE supports it (close the connection) but is clunkier than WS for it.

---

## D-004: TextIteratorStreamer with a thread

**Status:** Accepted, with a known footgun

**Context.** HuggingFace's `generate()` is blocking. To stream tokens out, you have to run generation in a thread and pipe output through a queue.

**Decision.** Use `TextIteratorStreamer` from transformers, run `generate()` in a background thread, push decoded chunks through an asyncio.Queue.

**Why.** This is the standard pattern. Works.

**Footgun.** PyTorch + threading + CUDA can deadlock if multiple requests hit the same model concurrently with default settings. We serialize requests with a single-instance lock at the server level. For multi-tenant scale, switch to vLLM as the backend.

---

## D-005: INT4 quantization for serving, BF16 for training

**Status:** Accepted

**Context.** Serving the model at INT4 fits in 8GB and stays fast. Training at INT4 with QLoRA also works but is slower and less stable.

**Decision.** BF16 + LoRA for training, INT4 (NF4) + LoRA for serving.

**Why.**
- Training quality is sensitive to quantization; weights matter.
- Inference quality is much less sensitive (you're not updating weights).
- Different runtime profiles for training vs serving anyway, so different precision is natural.

**Tradeoff.** You can't swap-in a freshly trained LoRA to a quantized base model and have it Just Work — the quantization changes the base weights' effective values. Re-train LoRA against the quantized base, or accept some quality drift. We accept the drift; it's measurable but small on VQA.

---

## D-006: VQAv2 grader exactly as the official benchmark

**Status:** Accepted

**Context.** Many implementations of VQA grading float around. They differ in subtle ways (article handling, punctuation, number formatting).

**Decision.** Implement the official grader: strip articles, strip punctuation, lowercase, normalize whitespace; score = min(matches/3, 1.0).

**Why.** Numbers compare across papers when the grader matches. Custom graders are a leading source of inflated benchmark claims.

---

## R-001: Reverted — ONNX export for serving

I started building an ONNX export path for the merged (base + LoRA) model. ONNXruntime would be faster than PyTorch for inference.

**Why I reverted.** Multimodal export in ONNX is rough — vision and text branches need different graphs, and the dynamic shapes in the visual feature path triggered ONNX edge cases. The win (~20% faster inference) wasn't worth the maintenance burden.

vLLM gives 2-3x speedups instead of 1.2x and handles VLMs cleanly. If serving speed matters, switch to vLLM, don't try to ONNX-ify.

## R-002: Reverted — Custom image patch tokenizer

Tried implementing my own image patch -> token pipeline for control. Worked for square images, broke on portrait/landscape extremes. LLaVA-NeXT's handling of variable aspect ratios is non-trivial. Used theirs and moved on.

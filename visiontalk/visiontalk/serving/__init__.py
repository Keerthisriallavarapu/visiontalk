"""HTTP API: image upload + question -> streamed answer.

Two endpoints:
- POST /ask        : returns the full answer once generation completes.
- POST /ask/stream : returns tokens as SSE events as they're generated.

Multipart upload for the image, JSON body for everything else.
"""
from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sse_starlette.sse import EventSourceResponse

from .models import VLMConfig, VLMWrapper

log = logging.getLogger(__name__)


def create_app(
    model_id: str | None = None,
    lora_path: str | Path | None = None,
    quantization: str | None = None,
) -> FastAPI:
    cfg = VLMConfig(
        model_id=model_id or "llava-hf/llava-v1.6-mistral-7b-hf",
        quantization=quantization,
    )
    vlm = VLMWrapper(cfg)
    if lora_path is not None and Path(lora_path).exists():
        vlm.attach_lora(lora_path)

    app = FastAPI(title="VisionTalk", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "model": cfg.model_id}

    @app.post("/ask")
    async def ask(
        image: UploadFile = File(...),
        question: str = Form(...),
        max_new_tokens: int = Form(256),
        temperature: float = Form(0.2),
    ):
        try:
            data = await image.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Bad image: {e}") from e

        try:
            text = vlm.generate(
                img, question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}") from e

        return {"answer": text}

    @app.post("/ask/stream")
    async def ask_stream(
        image: UploadFile = File(...),
        question: str = Form(...),
        max_new_tokens: int = Form(256),
        temperature: float = Form(0.2),
    ):
        data = await image.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")

        async def event_gen():
            # The streamer is a sync iterator; we yield it in a thread
            loop = asyncio.get_event_loop()
            q: asyncio.Queue = asyncio.Queue()

            def producer():
                try:
                    for chunk in vlm.generate_stream(
                        img, question,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    ):
                        asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
                except Exception as e:
                    log.exception("Stream producer failed: %s", e)
                finally:
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)

            loop.run_in_executor(None, producer)

            while True:
                chunk = await q.get()
                if chunk is None:
                    yield {"event": "done", "data": ""}
                    break
                yield {"event": "token", "data": chunk}

        return EventSourceResponse(event_gen())

    return app

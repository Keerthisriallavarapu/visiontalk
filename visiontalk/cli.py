"""CLI: train, eval, serve, run a single query."""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command(name="ask")
def ask_cmd(
    image: Path = typer.Argument(..., help="Path to image file."),
    question: str = typer.Argument(...),
    model_id: str = typer.Option("llava-hf/llava-v1.6-mistral-7b-hf"),
    quantization: str = typer.Option("", help="empty | int8 | int4"),
    lora: Path | None = typer.Option(None, help="Optional LoRA adapter path."),
):
    """Run a one-shot VQA query and print the answer."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    from .models import VLMConfig, VLMWrapper

    cfg = VLMConfig(model_id=model_id, quantization=quantization or None)
    vlm = VLMWrapper(cfg)
    if lora is not None:
        vlm.attach_lora(lora)

    console.print(f"[dim]Asking: {question}[/dim]")
    answer = vlm.generate(str(image), question)
    console.print()
    console.print(f"[bold]{answer}[/bold]")


@app.command(name="train")
def train_cmd(
    data: Path = typer.Argument(..., help="JSONL file with image_path/question/answer per line."),
    model_id: str = typer.Option("llava-hf/llava-v1.6-mistral-7b-hf"),
    out_dir: Path = typer.Option(Path("artifacts/lora")),
    epochs: int = 1,
    batch_size: int = 4,
    rank: int = 16,
):
    """Fine-tune the VLM with LoRA on a JSONL dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    from .training import LoRAConfig, TrainConfig, load_dataset_jsonl, train_lora

    items = load_dataset_jsonl(data)
    console.print(f"[green]Loaded {len(items)} training examples[/green]")

    final_path = train_lora(
        base_model_id=model_id,
        train_data=items,
        out_dir=out_dir,
        lora_config=LoRAConfig(r=rank),
        train_config=TrainConfig(n_epochs=epochs, batch_size=batch_size),
    )
    console.print(f"[green]LoRA saved to {final_path}[/green]")


@app.command(name="eval")
def eval_cmd(
    data: Path = typer.Argument(..., help="JSONL with image/question/answers."),
    model_id: str = typer.Option("llava-hf/llava-v1.6-mistral-7b-hf"),
    lora: Path | None = typer.Option(None),
    quantization: str = typer.Option(""),
    max_examples: int = typer.Option(500),
):
    """Run VQAv2-style eval on a JSONL dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    import json

    from PIL import Image as PILImage

    from .eval import evaluate_vqa_v2
    from .models import VLMConfig, VLMWrapper

    cfg = VLMConfig(model_id=model_id, quantization=quantization or None)
    vlm = VLMWrapper(cfg)
    if lora is not None:
        vlm.attach_lora(lora)

    examples = []
    with data.open() as f:
        for line in f:
            rec = json.loads(line)
            examples.append({
                "image": PILImage.open(rec["image_path"]).convert("RGB"),
                "question": rec["question"],
                "answers": rec.get("answers", [rec.get("answer", "")]),
                "question_type": rec.get("question_type", "all"),
            })

    result = evaluate_vqa_v2(vlm, examples, max_examples=max_examples)

    table = Table(title="VQA eval")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Examples", str(result.n_examples))
    table.add_row("Accuracy", f"{result.accuracy:.4f}")
    console.print(table)
    if result.by_category:
        console.print("\nBy category:")
        for cat, acc in sorted(result.by_category.items()):
            console.print(f"  {cat:.<30} {acc:.4f}")


@app.command(name="serve")
def serve_cmd(
    host: str = "0.0.0.0",
    port: int = 8090,
    model_id: str = typer.Option("llava-hf/llava-v1.6-mistral-7b-hf"),
    lora: Path | None = typer.Option(None),
    quantization: str = typer.Option(""),
):
    """Start the HTTP API."""
    import uvicorn

    from .serving import create_app

    logging.basicConfig(level=logging.INFO)
    app_ = create_app(
        model_id=model_id,
        lora_path=str(lora) if lora else None,
        quantization=quantization or None,
    )
    uvicorn.run(app_, host=host, port=port)


if __name__ == "__main__":
    app()

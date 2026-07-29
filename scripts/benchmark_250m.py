"""Benchmark bounded forward/backward cases for the 250M configuration."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from src.config import load_config
from src.model import LanguageModel, count_parameters


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    sequence_length: int
    micro_batch_size: int
    accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_backend: str | None = None
    optimizer_step: bool = False


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    status: str
    sequence_length: int
    micro_batch_size: int
    accumulation_steps: int
    gradient_checkpointing: bool
    compiled: bool
    compile_backend: str | None
    optimizer_step: bool
    tokens: int
    elapsed_seconds: float | None
    tokens_per_second: float | None
    peak_allocated_vram_bytes: int | None
    peak_reserved_vram_bytes: int | None
    error: str | None


CASES = (
    BenchmarkCase("context-512", 512, 1),
    BenchmarkCase("context-1024", 1024, 1),
    BenchmarkCase("context-2048", 2048, 1),
    BenchmarkCase("batch-2-context-512", 512, 2),
    BenchmarkCase("accumulation-2-context-512", 512, 1, accumulation_steps=2),
    BenchmarkCase(
        "checkpointed-context-2048",
        2048,
        1,
        gradient_checkpointing=True,
    ),
    BenchmarkCase(
        "optimizer-context-2048",
        2048,
        1,
        optimizer_step=True,
    ),
    BenchmarkCase(
        "optimizer-batch-2-context-2048",
        2048,
        2,
        optimizer_step=True,
    ),
    BenchmarkCase(
        "compiled-context-512",
        512,
        1,
        compile_model=True,
        compile_backend="aot_eager",
    ),
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                value,
                output_file,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def benchmark_case(
    model: LanguageModel,
    case: BenchmarkCase,
    *,
    vocab_size: int,
    device: torch.device,
) -> BenchmarkResult:
    model.set_gradient_checkpointing(case.gradient_checkpointing)
    executable: torch.nn.Module = model
    if case.compile_model:
        executable = torch.compile(model, backend=case.compile_backend)
    model.train()
    model.zero_grad(set_to_none=True)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=3e-4)
        if case.optimizer_step
        else None
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    token_total = (
        case.sequence_length
        * case.micro_batch_size
        * case.accumulation_steps
    )
    try:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for micro_step in range(case.accumulation_steps):
            generator = torch.Generator(device=device).manual_seed(
                42 + micro_step
            )
            input_ids = torch.randint(
                0,
                vocab_size,
                (case.micro_batch_size, case.sequence_length),
                device=device,
                generator=generator,
            )
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _logits, loss = executable(input_ids, labels)
                if loss is None:
                    raise RuntimeError("Model did not return a benchmark loss.")
                scaled_loss = loss / case.accumulation_steps
            scaled_loss.backward()
        if optimizer is not None:
            optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        result = BenchmarkResult(
            name=case.name,
            status="passed",
            sequence_length=case.sequence_length,
            micro_batch_size=case.micro_batch_size,
            accumulation_steps=case.accumulation_steps,
            gradient_checkpointing=case.gradient_checkpointing,
            compiled=case.compile_model,
            compile_backend=case.compile_backend,
            optimizer_step=case.optimizer_step,
            tokens=token_total,
            elapsed_seconds=elapsed,
            tokens_per_second=token_total / elapsed,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            error=None,
        )
    except torch.cuda.OutOfMemoryError as error:
        result = BenchmarkResult(
            name=case.name,
            status="out-of-memory",
            sequence_length=case.sequence_length,
            micro_batch_size=case.micro_batch_size,
            accumulation_steps=case.accumulation_steps,
            gradient_checkpointing=case.gradient_checkpointing,
            compiled=case.compile_model,
            compile_backend=case.compile_backend,
            optimizer_step=case.optimizer_step,
            tokens=token_total,
            elapsed_seconds=None,
            tokens_per_second=None,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            error=str(error),
        )
    except Exception as error:
        result = BenchmarkResult(
            name=case.name,
            status="failed",
            sequence_length=case.sequence_length,
            micro_batch_size=case.micro_batch_size,
            accumulation_steps=case.accumulation_steps,
            gradient_checkpointing=case.gradient_checkpointing,
            compiled=case.compile_model,
            compile_backend=case.compile_backend,
            optimizer_step=case.optimizer_step,
            tokens=token_total,
            elapsed_seconds=None,
            tokens_per_second=None,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        model.zero_grad(set_to_none=True)
        del optimizer
        del executable
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/250m.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/phase13/benchmark.json"),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in CASES),
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 13 benchmark requires CUDA.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 13 benchmark requires CUDA BF16.")

    config = load_config(arguments.config)
    device = torch.device("cuda")
    model = LanguageModel(config.model).to(device)
    selected = (
        CASES
        if arguments.case is None
        else tuple(case for case in CASES if case.name in arguments.case)
    )
    results: list[BenchmarkResult] = []
    for case in selected:
        print(f"Benchmarking {case.name}...", flush=True)
        result = benchmark_case(
            model,
            case,
            vocab_size=config.model.vocab_size,
            device=device,
        )
        results.append(result)
        print(
            f"{case.name}: {result.status}, "
            f"{result.tokens_per_second or 0:.1f} tokens/s",
            flush=True,
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(
            device
        ).total_memory,
        "config_path": str(arguments.config),
        "model_config": asdict(config.model),
        "parameter_count": count_parameters(model),
        "precision": "bf16",
        "results": [asdict(result) for result in results],
    }
    _atomic_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

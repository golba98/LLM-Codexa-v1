"""Display a compact live dashboard for the Codexa chat-v3 training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


WIDTH = 46
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[38;5;81m"
PURPLE = "\033[38;5;141m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;221m"


def latest_metrics(metrics_path: Path) -> dict[str, object]:
    """Read the most recent complete JSONL metric."""

    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def latest_validation(metrics_path: Path) -> dict[str, object] | None:
    """Return the most recent metric containing validation results."""

    for line in reversed(metrics_path.read_text(encoding="utf-8").splitlines()):
        value = json.loads(line)
        if value.get("validation_loss") is not None:
            return value
    return None


def gpu_status() -> tuple[str, str, str]:
    """Return VRAM, utilization, and temperature from nvidia-smi."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown", "unknown", "unknown"
    used, total, utilization, temperature = (
        item.strip() for item in result.stdout.strip().split(",")
    )
    return f"{used}/{total} MiB", f"{utilization}%", f"{temperature} C"


def render(metrics_path: Path, max_steps: int) -> None:
    """Render one dashboard frame."""

    metric = latest_metrics(metrics_path)
    validation = latest_validation(metrics_path)
    step = int(metric["optimizer_step"])
    ratio = min(step / max_steps, 1.0)
    filled = round(WIDTH * ratio)
    bar = f"{PURPLE}{'━' * filled}{DIM}{'─' * (WIDTH - filled)}{RESET}"
    remaining = max(max_steps - step, 0)
    step_seconds = float(metric["step_time_seconds"])
    eta_minutes = remaining * step_seconds / 60
    vram, gpu_use, temperature = gpu_status()
    validation_loss = "waiting"
    perplexity = "waiting"
    validation_step = "-"
    if validation is not None:
        validation_loss = f"{float(validation['validation_loss']):.4f}"
        perplexity = f"{float(validation['validation_perplexity']):.2f}"
        validation_step = str(validation["optimizer_step"])

    print("\033[2J\033[H", end="")
    print(f"{BOLD}{CYAN}CODEXA v1{RESET}  {DIM}CHAT PROTOCOL 3.0 · 920M SFT{RESET}")
    print(f"{DIM}{'─' * 64}{RESET}\n")
    print(f"  {bar}  {BOLD}{ratio * 100:5.1f}%{RESET}")
    print(f"  Step {BOLD}{step:,}{RESET} / {max_steps:,}"
          f"                         ETA {YELLOW}{eta_minutes:4.1f} min{RESET}\n")
    print(f"  Training loss      {GREEN}{float(metric['training_loss']):8.4f}{RESET}")
    print(f"  Validation loss    {CYAN}{validation_loss:>8}{RESET}  {DIM}(step {validation_step}){RESET}")
    print(f"  Validation ppl     {CYAN}{perplexity:>8}{RESET}")
    print(f"  Learning rate      {float(metric['learning_rate']):8.2e}")
    print(f"  Tokens seen        {int(metric['total_tokens_seen']):8,}")
    print(f"  Throughput         {float(metric['tokens_per_second']):8.0f} tok/s\n")
    print(f"  GPU utilization    {PURPLE}{gpu_use:>8}{RESET}")
    print(f"  GPU memory         {vram:>18}")
    print(f"  GPU temperature    {temperature:>8}\n")
    print(f"{DIM}  Refreshing every 2 seconds · Ctrl+C closes only this dashboard{RESET}")


def main() -> None:
    """Refresh until interrupted; training is managed by another process."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("logs/codexa-1b-chat-v3/train_metrics.jsonl"),
    )
    parser.add_argument("--max-steps", type=int, default=1_000)
    arguments = parser.parse_args()
    if arguments.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    try:
        while True:
            if arguments.metrics.is_file():
                render(arguments.metrics, arguments.max_steps)
            else:
                print("\033[2J\033[H", end="")
                print(f"{BOLD}{CYAN}CODEXA v1{RESET}")
                print(f"\n{DIM}Waiting for {arguments.metrics} ...{RESET}")
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

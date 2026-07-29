"""JSON-serializable NVIDIA GPU monitoring helpers."""

from datetime import datetime, timezone
import subprocess


GPU_QUERY_FIELDS = (
    "name",
    "temperature.gpu",
    "power.draw",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "clocks.sm",
    "clocks.mem",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_nvidia_smi_row(row: str) -> dict[str, object]:
    """Parse one no-header, no-units NVIDIA-SMI query row."""

    values = [value.strip() for value in row.strip().split(",")]
    if len(values) != len(GPU_QUERY_FIELDS):
        raise ValueError(
            f"Expected {len(GPU_QUERY_FIELDS)} GPU fields, got {len(values)}."
        )
    try:
        return {
            "timestamp_utc": utc_timestamp(),
            "gpu_name": values[0],
            "temperature_celsius": int(values[1]),
            "power_draw_watts": float(values[2]),
            "memory_used_mib": int(values[3]),
            "memory_total_mib": int(values[4]),
            "utilization_percent": int(values[5]),
            "sm_clock_mhz": int(values[6]),
            "memory_clock_mhz": int(values[7]),
        }
    except ValueError as error:
        raise ValueError(f"Malformed NVIDIA-SMI row: {row!r}") from error


def query_gpu() -> dict[str, object]:
    """Query the first visible NVIDIA GPU."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one visible NVIDIA GPU, got {len(rows)}."
        )
    return parse_nvidia_smi_row(rows[0])

"""Memory management and device helpers for 4GB VRAM constraint."""

import gc
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import torch

logger = logging.getLogger(__name__)
T = TypeVar("T")


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device preference ('auto', 'cuda', 'cpu') to torch.device.

    If 'auto' or 'cuda' requested but CUDA is not available, falls back to CPU.
    """
    preference = preference.lower().strip()
    if preference in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if preference == "cuda":
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def get_gpu_memory() -> dict[str, float]:
    """Return total, allocated, and free GPU memory in megabytes (MB)."""
    if not torch.cuda.is_available():
        return {"total_mb": 0.0, "allocated_mb": 0.0, "free_mb": 0.0}

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    allocated_bytes = torch.cuda.memory_allocated(0)
    reserved_bytes = torch.cuda.memory_reserved(0)

    total_mb = total_bytes / (1024 * 1024)
    allocated_mb = allocated_bytes / (1024 * 1024)
    free_mb = (total_bytes - reserved_bytes) / (1024 * 1024)

    return {
        "total_mb": round(total_mb, 2),
        "allocated_mb": round(allocated_mb, 2),
        "free_mb": round(free_mb, 2),
    }


def unload_model(model: Any) -> None:
    """Explicitly delete model, clear CUDA cache, and run garbage collection.

    Crucial for fitting inside 4 GB VRAM by running only one model at a time.
    """
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def oom_retry(
    fn: Callable[..., T],
    *args: Any,
    batch_size: int = 32,
    min_batch_size: int = 1,
    shrink_factor: int = 2,
    **kwargs: Any,
) -> T:
    """Execute a function with adaptive batch size reduction on CUDA OutOfMemoryError.

    If even batch_size=1 runs out of memory, retries on CPU with a warning.
    """
    current_batch_size = batch_size
    while current_batch_size >= min_batch_size:
        try:
            return fn(*args, batch_size=current_batch_size, **kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "CUDA OOM with batch size %d. Attempting recovery...", current_batch_size
            )
            unload_model(None)
            next_batch_size = current_batch_size // shrink_factor
            if next_batch_size < min_batch_size and current_batch_size == min_batch_size:
                logger.error(
                    "CUDA OOM at minimum batch size %d. Falling back to CPU.", min_batch_size
                )
                if "device" in kwargs:
                    kwargs["device"] = torch.device("cpu")
                return fn(*args, batch_size=1, **kwargs)
            current_batch_size = max(min_batch_size, next_batch_size)

    raise RuntimeError("Execution failed due to unrecoverable memory constraints.")

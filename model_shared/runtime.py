
from __future__ import annotations

import contextlib
import gc
from typing import Iterator

def _vram_stats() -> tuple[float, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.memory_reserved() / 1024 ** 2,
        )
    except ImportError:
        return 0.0, 0.0

def release_gpu_memory(*, verbose: bool = True) -> None:
    try:
        import torch
    except ImportError:
        if verbose:
            print("  release_gpu_memory: torch not available, skipping.")
        return

    before_alloc, before_reserved = _vram_stats()

    for _ in range(3):
        gc.collect()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        after_alloc, after_reserved = _vram_stats()
        if verbose:
            print(
                f"  release_gpu_memory: "
                f"allocated {before_alloc:.0f} → {after_alloc:.0f} MB, "
                f"reserved {before_reserved:.0f} → {after_reserved:.0f} MB",
                flush=True,
            )
    elif verbose:
        print("  release_gpu_memory: no CUDA device.", flush=True)

@contextlib.contextmanager
def gpu_cleanup_on_exit(*, verbose: bool = True) -> Iterator[None]:
    interrupted = False
    failed = False
    try:
        yield
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("\n⏹  training interrupted by user; releasing GPU VRAM …", flush=True)
        raise
    except Exception:
        failed = True
        if verbose:
            print("\n💥 training raised an exception; releasing GPU VRAM …", flush=True)
        raise
    finally:
        # Two-stage cleanup: first while we're still inside the trainer
        # function's frame (best-effort), then re-run via the trainer's own
        # `finally` once that frame has died. Empty cache on normal exit too
        # so back-to-back stages don't accumulate fragmentation.
        release_gpu_memory(verbose=verbose and (interrupted or failed))

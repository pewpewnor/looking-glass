
from __future__ import annotations

import os
import random as _random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_DRIVE_PREFIXES = ("/content/drive/", "/content/gdrive/")

def _is_drive_path(p: Path) -> bool:
    s = str(p.resolve()) if p.exists() else str(p)
    return any(s.startswith(prefix) for prefix in _DRIVE_PREFIXES)

def is_drive_path(p: str | Path) -> bool:
    return _is_drive_path(Path(p))

def audit_drive_directory(
    out_dir: str | Path, *, expected_globs: tuple[str, ...]
) -> dict:
    out_dir = Path(out_dir)
    present: list[str] = []
    missing: list[str] = []
    for pat in expected_globs:
        matches = sorted(p.name for p in out_dir.glob(pat))
        if matches:
            present.extend(matches)
        else:
            missing.append(pat)
    return {"out_dir": str(out_dir), "present": present, "missing": missing}

def assert_checkpoint_root_on_drive(out_root: str | Path, *, on_colab: bool) -> None:
    out_root = Path(out_root)
    if on_colab and not _is_drive_path(out_root):
        raise RuntimeError(
            f"Colab runtime detected, but checkpoint OUT_ROOT={out_root} "
            f"does not live on Google Drive (expected one of "
            f"{_DRIVE_PREFIXES}). Checkpoints written to the runtime SSD "
            f"will be LOST on runtime reset. Repoint OUT_ROOT into your "
            f"Drive-mounted project root."
        )

def capture_rng() -> dict:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": _random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng(state: dict | None) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        _random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # fsync on a directory is unsupported on some filesystems (incl. some
        # FUSE mounts). The file-level fsync above is the durability anchor;
        # the directory fsync is purely belt-and-suspenders.
        pass

def _torch_save_with_fsync(obj: Any, tmp_path: Path) -> None:
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some FUSE drivers throw ENOTSUP / EINVAL on fsync.  In that
            # case we have to trust the close() to do the right thing.
            pass

def _local_scratch_dir() -> Path:
    base = Path(os.environ.get("CKPT_SCRATCH", "/tmp")) / "ckpt_scratch"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _save_local_then_copy(
    obj: Any,
    dest: Path,
    *,
    max_retries: int,
    retry_backoff_s: float,
) -> None:
    scratch_dir = _local_scratch_dir()
    pid = os.getpid()
    fd, tmp_local_str = tempfile.mkstemp(
        suffix=".pt",
        prefix=f"ckpt_{pid}_",
        dir=str(scratch_dir),
    )
    os.close(fd)
    tmp_local = Path(tmp_local_str)
    try:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                _torch_save_with_fsync(obj, tmp_local)
                last_err = None
                break
            except OSError as e:
                last_err = e
                if attempt + 1 < max_retries:
                    time.sleep(retry_backoff_s * (2**attempt))
        if last_err is not None:
            raise last_err
        _copy_local_to_drive(
            tmp_local,
            dest,
            expected_size=tmp_local.stat().st_size,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
    finally:
        try:
            if tmp_local.exists():
                tmp_local.unlink()
        except OSError:
            pass

def _save_inplace_atomic(
    obj: Any,
    path: Path,
    *,
    max_retries: int,
    retry_backoff_s: float,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            _torch_save_with_fsync(obj, tmp)
            os.replace(str(tmp), str(path))
            _fsync_dir(path.parent)
            last_err = None
            break
        except OSError as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt + 1 < max_retries:
                wait = retry_backoff_s * (2**attempt)
                print(
                    f"  ⚠ atomic_save: transient OSError on {path} "
                    f"(attempt {attempt + 1}/{max_retries}, retrying in {wait:.1f}s): {e}",
                    flush=True,
                )
                time.sleep(wait)
    if last_err is not None:
        raise last_err

def atomic_save(
    obj: Any,
    path: Path,
    *,
    quiet: bool = False,
    label: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if _is_drive_path(path):
        _save_local_then_copy(
            obj,
            path,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
    else:
        _save_inplace_atomic(
            obj,
            path,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )

    if not quiet:
        try:
            mb = path.stat().st_size / (1024 * 1024)
            size_str = f"  ({mb:.1f} MB)"
        except OSError:
            size_str = ""
        prefix = f"[{label}] " if label else ""
        on_drive = " [Drive]" if _is_drive_path(path) else ""
        print(f"  ✓ {prefix}saved checkpoint{on_drive}: {path}{size_str}", flush=True)

def atomic_save_multi(
    obj: Any,
    targets: list[tuple[Path, str]],
    *,
    quiet: bool = False,
    max_retries: int = 3,
    retry_backoff_s: float = 1.5,
) -> None:
    if not targets:
        return

    for dest, _ in targets:
        dest.parent.mkdir(parents=True, exist_ok=True)

    drive_targets = [(d, l) for d, l in targets if _is_drive_path(d)]
    local_targets = [(d, l) for d, l in targets if not _is_drive_path(d)]

    if drive_targets:
        # Stage once on local SSD, then copy to each Drive destination.
        scratch_dir = _local_scratch_dir()
        pid = os.getpid()
        fd, tmp_local_str = tempfile.mkstemp(
            suffix=".pt",
            prefix=f"ckpt_multi_{pid}_",
            dir=str(scratch_dir),
        )
        os.close(fd)
        tmp_local = Path(tmp_local_str)
        try:
            last_err: Exception | None = None
            for attempt in range(max_retries):
                try:
                    _torch_save_with_fsync(obj, tmp_local)
                    last_err = None
                    break
                except OSError as e:
                    last_err = e
                    if attempt + 1 < max_retries:
                        time.sleep(retry_backoff_s * (2**attempt))
            if last_err is not None:
                raise last_err
            staged_size = tmp_local.stat().st_size

            for dest, label in drive_targets:
                _copy_local_to_drive(
                    tmp_local,
                    dest,
                    expected_size=staged_size,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                )
                if not quiet:
                    try:
                        mb = dest.stat().st_size / (1024 * 1024)
                        size_str = f"  ({mb:.1f} MB)"
                    except OSError:
                        size_str = ""
                    prefix = f"[{label}] " if label else ""
                    print(
                        f"  ✓ {prefix}saved checkpoint [Drive]: {dest}{size_str}",
                        flush=True,
                    )
        finally:
            try:
                if tmp_local.exists():
                    tmp_local.unlink()
            except OSError:
                pass

    if local_targets:
        # Local writes: do each one separately; they're cheap and reliable.
        for dest, label in local_targets:
            _save_inplace_atomic(
                obj,
                dest,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )
            if not quiet:
                try:
                    mb = dest.stat().st_size / (1024 * 1024)
                    size_str = f"  ({mb:.1f} MB)"
                except OSError:
                    size_str = ""
                prefix = f"[{label}] " if label else ""
                print(
                    f"  ✓ {prefix}saved checkpoint: {dest}{size_str}",
                    flush=True,
                )

    missing = [str(d) for d, _ in targets if not d.exists()]
    if missing:
        raise RuntimeError(
            f"atomic_save_multi: post-save verification failed; "
            f"these targets do not exist on disk: {missing}"
        )

def _copy_local_to_drive(
    src_local: Path,
    dest: Path,
    *,
    expected_size: int,
    max_retries: int,
    retry_backoff_s: float,
) -> None:
    pid = os.getpid()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        # Each retry uses a fresh partial name Drive has never seen.
        unique = f"{pid}.{time.time_ns()}.{attempt}"
        partial = dest.with_suffix(dest.suffix + f".partial.{unique}")
        try:
            shutil.copy2(str(src_local), str(partial))
            try:
                fdp = os.open(str(partial), os.O_RDONLY)
                try:
                    os.fsync(fdp)
                finally:
                    os.close(fdp)
            except OSError:
                pass
            partial_size = partial.stat().st_size
            if partial_size != expected_size:
                raise OSError(
                    f"short write to Drive: partial size {partial_size} "
                    f"!= expected {expected_size}"
                )
            # Delete-then-rename to avoid Drive's rename-over-existing edge cases.
            try:
                if dest.exists():
                    dest.unlink()
            except OSError:
                pass
            os.replace(str(partial), str(dest))
            _fsync_dir(dest.parent)
            final_size = dest.stat().st_size
            if final_size != expected_size:
                raise OSError(
                    f"size mismatch after rename: {final_size} != {expected_size}"
                )
            last_err = None
            break
        except OSError as e:
            last_err = e
            try:
                if partial.exists():
                    partial.unlink()
            except OSError:
                pass
            if attempt + 1 < max_retries:
                wait = retry_backoff_s * (2**attempt)
                print(
                    f"  ⚠ atomic_save: Drive copy of {dest} failed "
                    f"(attempt {attempt + 1}/{max_retries}, retrying in {wait:.1f}s): {e}",
                    flush=True,
                )
                time.sleep(wait)
    if last_err is not None:
        raise last_err

def resolve_resume_path(resume: bool | str | None, out_dir: Path) -> Path | None:
    if resume is False or resume is None:
        return None
    if resume is True:
        p = out_dir / "last.pt"
        return p if p.exists() else None
    p = Path(resume)
    if not p.is_absolute():
        p = out_dir / p
    return p if p.exists() else None

def hygiene(out_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    rolling = sorted(
        list(out_dir.glob("ckpt_fold*_epoch*.pt")),
        key=lambda p: p.stat().st_mtime,
    )
    if len(rolling) <= keep_last_n:
        return
    for p in rolling[: len(rolling) - keep_last_n]:
        try:
            p.unlink()
        except OSError:
            pass

def quarantine_incompatible(out_dir: Path, reason: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = out_dir / f"legacy_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for p in list(out_dir.glob("*.pt")):
        try:
            os.replace(str(p), str(backup / p.name))
            moved.append(p.name)
        except OSError:
            pass
    print(
        f"⚠ Quarantined {len(moved)} incompatible ckpt(s) → {backup}\n  reason: {reason}"
    )
    return backup

def get_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    sd = model.state_dict()
    trainable_param_names = {n for n, p in model.named_parameters() if p.requires_grad}
    for name in trainable_param_names:
        if name in sd:
            out[name] = sd[name].detach().cpu()
    trainable_module_prefixes = set()
    for name in trainable_param_names:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            trainable_module_prefixes.add(".".join(parts[: i - 1]))
    for buf_name, buf in model.named_buffers():
        prefix = ".".join(buf_name.split(".")[:-1])
        if prefix in trainable_module_prefixes:
            out[buf_name] = buf.detach().cpu()
    return out

def load_trainable_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    own = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape != v.shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}")
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        head = "; ".join(skipped[:5])
        tail = f" (and {len(skipped) - 5} more)" if len(skipped) > 5 else ""
        print(
            f"⚠ load_trainable_state skipped {len(skipped)} mismatched keys: {head}{tail}"
        )
    return list(missing), list(unexpected)

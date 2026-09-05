
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 6

def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    p = Path(manifest_path)
    with open(p) as f:
        m = json.load(f)
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version mismatch (got {m.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}). Re-run scripts/aggregator.py."
        )
    return m

def filter_instances(
    manifest: dict[str, Any],
    *,
    split: str | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    out = list(manifest["instances"])
    if split is not None:
        out = [i for i in out if i.get("split") == split]
    if sources is not None:
        ssrc = set(sources)
        out = [i for i in out if i.get("source") in ssrc]
    return out

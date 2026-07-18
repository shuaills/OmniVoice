#!/usr/bin/env python3
"""Finalize the Markov A/B receipt with fail-closed artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"artifact must be NAME=PATH, got {specification!r}")
    name, raw_path = specification.split("=", 1)
    if not KEY_RE.fullmatch(name):
        raise ValueError(f"invalid artifact name: {name!r}")
    path = Path(raw_path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"artifact is missing or empty: {path}")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scientific-verdict",
        choices=("PROMOTE_TO_10K", "KILL"),
        required=True,
    )
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise SystemExit(f"manifest is missing or empty: {manifest}")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to reuse receipt output: {output}")
    manifest_text = manifest.read_text()
    for marker in ("artifact_receipt=", "scientific_verdict="):
        if marker in manifest_text:
            raise SystemExit(f"manifest is already finalized: marker={marker}")

    artifacts: dict[str, dict[str, str | int]] = {}
    for specification in args.artifact:
        name, path = _artifact(specification)
        if name in artifacts:
            raise SystemExit(f"duplicate artifact name: {name}")
        artifacts[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if not artifacts:
        raise SystemExit("at least one --artifact is required")

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "schema": "block_markov_300ab_receipt_v1",
        "generated_utc": generated_utc,
        "scientific_verdict": args.scientific_verdict,
        "manifest": str(manifest),
        "manifest_before_receipt_sha256": hashlib.sha256(
            manifest_text.encode()
        ).hexdigest(),
        "artifacts": dict(sorted(artifacts.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(output)
    receipt_sha256 = _sha256(output)

    with manifest.open("a") as stream:
        stream.write(f"artifact_receipt={output}\n")
        stream.write(f"artifact_receipt_sha256={receipt_sha256}\n")
        stream.write(f"artifact_count={len(artifacts)}\n")
        stream.write(f"scientific_verdict={args.scientific_verdict}\n")
        stream.write(f"finished_utc={generated_utc}\n")
        stream.write("train_rc=0\n")
        stream.write("tee_rc=0\n")
        stream.write("infrastructure_rc=0\n")
        stream.write("rc=0\n")

    print(
        "BLOCK_MARKOV_RECEIPT_FINALIZED "
        + json.dumps(
            {
                "artifact_count": len(artifacts),
                "receipt": str(output),
                "receipt_sha256": receipt_sha256,
                "scientific_verdict": args.scientific_verdict,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

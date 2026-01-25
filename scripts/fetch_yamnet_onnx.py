#!/usr/bin/env python3
"""Download YamNet ONNX model into services/worker_features/yamnet_onnx.

The script fetches the model from a pinned URL, verifies the sha256 recorded
in services/worker_features/yamnet_onnx/yamnet.onnx.sha256, and writes the
artifact as yamnet.onnx. The default URL points at a future GitHub Release
asset; replace it (or export YAMNET_ONNX_DOWNLOAD_URL) once the asset is
published.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YAMNET_DIR = REPO_ROOT / "services" / "worker_features" / "yamnet_onnx"
TARGET_PATH = YAMNET_DIR / "yamnet.onnx"
SHA256_PATH = YAMNET_DIR / "yamnet.onnx.sha256"
DEFAULT_URL = (
    "https://github.com/SOVRN144/ssis-audio-pipeline-assets/releases/download/yamnet-v1/yamnet.onnx"
)


def read_expected_hash() -> str:
    if not SHA256_PATH.exists():
        raise SystemExit(
            f"SHA256 file missing: {SHA256_PATH}. Please add the correct hash before fetching."
        )
    text = SHA256_PATH.read_text().strip()
    if not text:
        raise SystemExit(
            f"SHA256 file {SHA256_PATH} is empty; update it with the real yamnet.onnx hash."
        )
    if "placeholder" in text.lower():
        raise SystemExit(
            textwrap.dedent(
                f"""
                The file {SHA256_PATH} still contains a placeholder hash.
                Publish the official yamnet.onnx asset and update the file with
                its 64-character sha256 before running this script.
                """
            ).strip()
        )
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text.lower()):
        raise SystemExit(f"Expected a 64-character hex hash in {SHA256_PATH}, got '{text}'.")
    return text.lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch yamnet.onnx with sha verification")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite yamnet.onnx even if it already exists",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TARGET_PATH,
        help=f"Destination path (default: {TARGET_PATH})",
    )
    args = parser.parse_args()

    expected_hash = read_expected_hash()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        current = sha256_file(output_path)
        if current == expected_hash:
            print(f"yamnet.onnx already present at {output_path}")
            return
        print(
            "Existing yamnet.onnx hash does not match expected value; use --force to replace.",
            file=sys.stderr,
        )
        return

    url = os.environ.get("YAMNET_ONNX_DOWNLOAD_URL", DEFAULT_URL)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_fd)
    tmp_file = Path(tmp_path)
    try:
        print(f"Downloading yamnet.onnx from {url} ...")
        download(url, tmp_file)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network failures
        tmp_file.unlink(missing_ok=True)
        raise SystemExit(
            f"Failed to download yamnet.onnx ({exc.code} {exc.reason}): {url}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network failures
        tmp_file.unlink(missing_ok=True)
        raise SystemExit(f"Failed to download yamnet.onnx: {exc}") from exc

    actual_hash = sha256_file(tmp_file)
    if actual_hash != expected_hash:
        tmp_file.unlink(missing_ok=True)
        raise SystemExit(
            "Hash mismatch for downloaded yamnet.onnx (expected "
            f"{expected_hash}, got {actual_hash})."
        )

    shutil.move(str(tmp_file), output_path)
    print(f"Saved yamnet.onnx to {output_path}")


if __name__ == "__main__":
    main()

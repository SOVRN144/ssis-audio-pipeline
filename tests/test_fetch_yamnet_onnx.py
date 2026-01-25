from __future__ import annotations

import io
from pathlib import Path

import pytest

import scripts.fetch_yamnet_onnx as fy


def _set_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    sha_path = tmp_path / "yamnet.onnx.sha256"
    out_path = tmp_path / "yamnet.onnx"
    monkeypatch.setattr(fy, "SHA256_PATH", sha_path)
    monkeypatch.setattr(fy, "TARGET_PATH", out_path)
    return sha_path, out_path


def test_read_expected_hash_rejects_empty(tmp_path, monkeypatch):
    sha_path, _ = _set_paths(monkeypatch, tmp_path)
    sha_path.write_text("")

    with pytest.raises(SystemExit):
        fy.read_expected_hash()


def test_read_expected_hash_rejects_placeholder(tmp_path, monkeypatch):
    sha_path, _ = _set_paths(monkeypatch, tmp_path)
    sha_path.write_text("PLACEHOLDER_HASH")

    with pytest.raises(SystemExit):
        fy.read_expected_hash()


def test_read_expected_hash_rejects_invalid_hex(tmp_path, monkeypatch):
    sha_path, _ = _set_paths(monkeypatch, tmp_path)
    sha_path.write_text("g" * 64)

    with pytest.raises(SystemExit):
        fy.read_expected_hash()


def test_main_exits_nonzero_on_existing_hash_mismatch(tmp_path, monkeypatch, capsys):
    sha_path, out_path = _set_paths(monkeypatch, tmp_path)
    out_path.write_bytes(b"abc")
    sha_path.write_text("0" * 64)

    result = fy.main(["--output", str(out_path)])

    captured = capsys.readouterr()
    assert "hash does not match" in captured.err
    assert result == 1


def test_main_returns_zero_when_existing_hash_matches(tmp_path, monkeypatch):
    sha_path, out_path = _set_paths(monkeypatch, tmp_path)
    data = b"fixture-bytes"
    out_path.write_bytes(data)
    sha_path.write_text(fy.sha256_file(out_path))

    result = fy.main(["--output", str(out_path)])

    assert result == 0


def test_download_passes_timeout(tmp_path, monkeypatch):
    written = tmp_path / "downloaded.onnx"

    class FakeResp(io.BytesIO):
        def __enter__(self):  # pragma: no cover - trivial
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
            return False

    def fake_urlopen(url, timeout=None):
        assert timeout == 12.34
        assert url == "http://example.invalid/yamnet.onnx"
        return FakeResp(b"abc123")

    monkeypatch.setattr(fy.urllib.request, "urlopen", fake_urlopen)

    fy.download("http://example.invalid/yamnet.onnx", written, timeout=12.34)

    assert written.read_bytes() == b"abc123"

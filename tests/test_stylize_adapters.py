"""Adapter plumbing tests — verify selection + output handling without network.

We monkeypatch each adapter's `_run()` (the only part that touches the provider)
and assert that stylize() correctly turns the provider's output into a written
image file. This exercises everything except the live HTTP call.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from woodcut.stylize import available_adapters, get_adapter
from woodcut.stylize.fal_adapter import FalAdapter, _first_image_url
from woodcut.stylize.replicate_adapter import ReplicateAdapter, _fetch_bytes


def _make_png(path: Path, color=(120, 90, 60)) -> Path:
    Image.new("RGB", (32, 24), color).save(path)
    return path


def test_registry_has_all_adapters():
    assert set(available_adapters()) >= {"stub", "replicate", "fal"}
    assert isinstance(get_adapter("replicate"), ReplicateAdapter)
    assert isinstance(get_adapter("fal"), FalAdapter)


def test_unknown_adapter_raises():
    with pytest.raises(KeyError):
        get_adapter("nope")


class _FakeFileOutput:
    """Mimics replicate>=0.25 FileOutput (has .read())."""
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def test_replicate_stylize_with_fileoutput(tmp_path, monkeypatch):
    src = _make_png(tmp_path / "photo.png")
    payload = src.read_bytes()
    adapter = ReplicateAdapter()
    monkeypatch.setattr(adapter, "_run", lambda *a, **k: _FakeFileOutput(payload))

    out = adapter.stylize(src, "prompt", tmp_path / "out.png")
    assert out.exists()
    assert Image.open(out).size == (32, 24)


def test_replicate_stylize_with_url_str(tmp_path, monkeypatch):
    src = _make_png(tmp_path / "photo.png", color=(10, 80, 80))
    adapter = ReplicateAdapter()
    # Return a file:// URL the stdlib fetcher can read (stands in for an https URL).
    monkeypatch.setattr(adapter, "_run", lambda *a, **k: src.resolve().as_uri())

    out = adapter.stylize(src, "prompt", tmp_path / "out.png")
    assert out.exists() and out.stat().st_size > 0


def test_replicate_fetch_bytes_rejects_garbage():
    with pytest.raises(RuntimeError):
        _fetch_bytes(object())


def test_fal_stylize_downloads_first_image(tmp_path, monkeypatch):
    src = _make_png(tmp_path / "photo.png", color=(60, 60, 90))
    adapter = FalAdapter()
    monkeypatch.setattr(
        adapter, "_run",
        lambda *a, **k: {"images": [{"url": src.resolve().as_uri()}]},
    )

    out = adapter.stylize(src, "prompt", tmp_path / "out.png")
    assert out.exists()
    assert Image.open(out).size == (32, 24)


def test_fal_result_parsing_validates():
    assert _first_image_url({"images": [{"url": "http://x/y.png"}]}) == "http://x/y.png"
    with pytest.raises(RuntimeError):
        _first_image_url({"images": []})

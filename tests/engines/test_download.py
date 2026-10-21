import hashlib
from pathlib import Path

import pytest

from velora.engines.base import EngineError
from velora.engines.download import RemoteFile, fetch


def remote_for(source: Path, *, sha256: str | None = None, size: int | None = None) -> RemoteFile:
    data = source.read_bytes()
    return RemoteFile(
        name="weights.bin",
        url=source.as_uri(),
        sha256=sha256 or hashlib.sha256(data).hexdigest(),
        size=len(data) if size is None else size,
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "src" / "weights.bin"
    path.parent.mkdir()
    path.write_bytes(b"x" * 3000)
    return path


def test_fetch_downloads_verifies_and_reports_progress(source: Path, tmp_path: Path) -> None:
    remote = remote_for(source)
    seen: list[tuple[str, int, int]] = []
    target = fetch(remote, tmp_path / "cache", lambda *args: seen.append(args))
    assert target.read_bytes() == source.read_bytes()
    assert remote.present(tmp_path / "cache")
    assert seen[-1] == ("weights.bin", 3000, 3000)
    assert not list((tmp_path / "cache").glob("*.part"))


def test_fetch_rejects_checksum_mismatch(source: Path, tmp_path: Path) -> None:
    remote = remote_for(source, sha256="0" * 64)
    with pytest.raises(EngineError, match="corrupt"):
        fetch(remote, tmp_path / "cache", lambda *_: None)
    assert not list((tmp_path / "cache").iterdir())


def test_fetch_reports_unreachable_url(tmp_path: Path) -> None:
    remote = RemoteFile("w.bin", (tmp_path / "missing.bin").as_uri(), "0" * 64, 1)
    with pytest.raises(EngineError, match=r"download of w\.bin failed"):
        fetch(remote, tmp_path / "cache", lambda *_: None)


def test_present_requires_matching_size(source: Path, tmp_path: Path) -> None:
    remote = remote_for(source, size=1)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"x" * 3000)
    assert not remote.present(cache)

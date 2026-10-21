"""Checksum-verified downloads of engine weight files."""

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from velora.config import DOWNLOAD_BLOCK_BYTES
from velora.engines.base import EngineError, Progress


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    sha256: str
    size: int

    def path(self, directory: Path) -> Path:
        return directory / self.name

    def present(self, directory: Path) -> bool:
        path = self.path(directory)
        return path.is_file() and path.stat().st_size == self.size


def fetch(remote: RemoteFile, directory: Path, progress: Progress) -> Path:
    """Download to a .part file, verify size and sha256, then move into place."""
    directory.mkdir(parents=True, exist_ok=True)
    target = remote.path(directory)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(remote.url) as response, partial.open("wb") as out:
            while block := response.read(DOWNLOAD_BLOCK_BYTES):
                out.write(block)
                digest.update(block)
                done += len(block)
                progress(remote.name, done, remote.size)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise EngineError(f"download of {remote.name} failed: {exc}") from exc

    if done != remote.size or digest.hexdigest() != remote.sha256:
        partial.unlink(missing_ok=True)
        raise EngineError(f"download of {remote.name} is corrupt (size or checksum mismatch)")
    partial.replace(target)
    return target

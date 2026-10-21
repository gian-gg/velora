"""Shared defaults and constants."""

import os
from pathlib import Path

DEFAULT_WPM = 150

UNTITLED_SNIPPET_CHARS = 40

DEFAULT_CHUNK_CHARS = 300

DOWNLOAD_BLOCK_BYTES = 1 << 20
KOKORO_DEFAULT_VOICE = "af_heart"
DEFAULT_ENGINE = "kokoro"
DEFAULT_OUTPUT_SUFFIX = ".m4a"

PAUSE_CHUNK_SECONDS = 0.25
PAUSE_PARAGRAPH_SECONDS = 0.7
AAC_BITRATE = "64k"
MP3_QUALITY = "4"

INSPECT_MAX_TITLE_WIDTH = 48
INSPECT_PREVIEW_PARAGRAPHS = 5
INSPECT_MIN_SHOWN_MINUTES = 0.5


def models_dir() -> Path:
    """Where engine weights are cached: $XDG_CACHE_HOME/velora/models."""
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "velora" / "models"

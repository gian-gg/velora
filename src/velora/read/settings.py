"""Settings that apply to reading with any engine."""

from velora.config import PAUSE_CHUNK_SECONDS, PAUSE_PARAGRAPH_SECONDS
from velora.engines.settings import Setting

READ_SETTINGS = (
    Setting("chunk_pause", PAUSE_CHUNK_SECONDS, "Silence between chunks, in seconds", 0.0, 5.0),
    Setting(
        "paragraph_pause",
        PAUSE_PARAGRAPH_SECONDS,
        "Silence after paragraphs and the chapter title, in seconds",
        0.0,
        5.0,
    ),
)

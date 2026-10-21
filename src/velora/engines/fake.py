"""A dependency-free engine that renders text as a quiet tone, for tests."""

import numpy as np

from velora.engines.base import Progress, Samples, Voice, WeightsStatus, reject_unknown_keys

SECONDS_PER_CHAR = 0.01
DEFAULT_FREQUENCY = 220.0
AMPLITUDE = 0.1


class FakeEngine:
    name = "fake"
    extra = ""
    requires: tuple[str, ...] = ()
    max_chunk_chars = 200
    sample_rate = 16_000

    def weights_status(self) -> WeightsStatus:
        return WeightsStatus(downloaded=True)

    def download(self, progress: Progress) -> None:
        progress("fake", 0, 0)

    def synthesize(self, text: str, voice: Voice) -> Samples:
        reject_unknown_keys(self, voice, {"frequency"})
        frequency = float(voice.get("frequency", DEFAULT_FREQUENCY))
        count = int(len(text) * SECONDS_PER_CHAR * self.sample_rate)
        t = np.arange(count, dtype=np.float32) / self.sample_rate
        return (AMPLITUDE * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

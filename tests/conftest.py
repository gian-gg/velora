from pathlib import Path

import pytest

from tests.epub_factory import MakeEpub, epub_builder


@pytest.fixture
def make_epub(tmp_path: Path) -> MakeEpub:
    return epub_builder(tmp_path)

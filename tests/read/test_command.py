import shutil
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.epub_factory import Doc, MakeEpub
from velora.cli import app

runner = CliRunner()
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def book(make_epub: MakeEpub) -> Path:
    return make_epub(
        [
            Doc("<p>For my readers.</p>", toc_title="Dedication"),
            Doc("<h1>Chapter 1</h1><p>It began. It went on.</p>", toc_title="Chapter 1"),
            Doc("<h1>Chapter 2</h1><p>It ended.</p>", toc_title="Chapter 2"),
        ],
        title="Tiny",
        author="A. Writer",
        name="tiny.epub",
    )


def seconds(path: Path) -> float:
    with wave.open(str(path)) as wav:
        return float(wav.getnframes() / wav.getframerate())


@needs_ffmpeg
def test_reads_one_chapter_with_fake_engine(book: Path, tmp_path: Path) -> None:
    out = tmp_path / "one.wav"
    result = runner.invoke(
        app, ["read", str(book), "--chapter", "2", "--engine", "fake", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "wrote" in result.stdout
    assert seconds(out) > 0


@needs_ffmpeg
def test_range_is_one_file_by_default_and_split_makes_one_per_chapter(
    book: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    combined = runner.invoke(app, ["read", str(book), "--chapters", "1-3", "--engine", "fake"])
    assert combined.exit_code == 0, combined.output
    assert (tmp_path / "tiny-ch01-03.m4a").is_file()

    folder = tmp_path / "split"
    split = runner.invoke(
        app,
        ["read", str(book), "--chapters", "2-3", "--engine", "fake", "--split", "-o", str(folder)],
    )
    assert split.exit_code == 0, split.output
    assert sorted(p.name for p in folder.iterdir()) == ["tiny-ch02.m4a", "tiny-ch03.m4a"]


def test_kokoro_without_weights_points_to_download(
    book: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    result = runner.invoke(
        app, ["read", str(book), "--chapter", "1", "-o", str(tmp_path / "x.m4a")]
    )
    assert result.exit_code == 1
    assert "Run: velora models download kokoro" in result.stderr
    assert not (tmp_path / "x.m4a").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--chapter", "9"], "between 1 and 3"),
        ([], "exactly one of --chapter"),
        (["--chapter", "1", "--engine", "nope"], "unknown engine 'nope'"),
        (
            ["--chapter", "1", "--engine", "fake", "-o", "/no/such/dir/x.wav"],
            "output folder does not exist",
        ),
    ],
)
def test_bad_requests_fail_before_synthesis(book: Path, args: list[str], message: str) -> None:
    result = runner.invoke(app, ["read", str(book), *args])
    assert result.exit_code == 1
    assert message in result.stderr


@needs_ffmpeg
def test_engine_error_mid_run_leaves_no_file(book: Path, tmp_path: Path) -> None:
    out = tmp_path / "x.wav"
    result = runner.invoke(
        app,
        ["read", str(book), "--chapter", "1", "--engine", "fake", "--voice", "x", "-o", str(out)],
    )
    assert result.exit_code == 1
    assert "does not support voice setting(s) voice" in result.stderr
    assert not list(tmp_path.glob("*.wav"))
    assert not list(tmp_path.glob(".*part*"))


def test_missing_book_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["read", str(tmp_path / "nope.epub"), "--chapter", "1"])
    assert result.exit_code == 1
    assert "file not found" in result.stderr

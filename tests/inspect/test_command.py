from pathlib import Path

from typer.testing import CliRunner

from tests.epub_factory import Doc, MakeEpub
from velora.cli import app

runner = CliRunner()


def two_chapter_book(make_epub: MakeEpub) -> Path:
    return make_epub(
        [
            Doc("<h1>Prologue</h1><p>It began.</p>", toc_title="Prologue"),
            Doc("<h1>One</h1>" + "<p>word word word</p>" * 100, toc_title="Chapter One"),
        ],
        title="Dune",
        author="Frank Herbert",
    )


def test_inspect_lists_chapters(make_epub: MakeEpub) -> None:
    result = runner.invoke(app, ["inspect", str(two_chapter_book(make_epub))])
    assert result.exit_code == 0
    assert result.stdout.startswith("Dune — Frank Herbert")
    assert "Prologue" in result.stdout
    assert "Chapter One" in result.stdout
    assert "2 chapters · 304 words · ~2m at 150 wpm" in result.stdout


def test_inspect_wpm_changes_estimate(make_epub: MakeEpub) -> None:
    result = runner.invoke(app, ["inspect", str(two_chapter_book(make_epub)), "--wpm", "304"])
    assert result.exit_code == 0
    assert "~1m at 304 wpm" in result.stdout


def test_inspect_show_previews_chapter(make_epub: MakeEpub) -> None:
    result = runner.invoke(app, ["inspect", str(two_chapter_book(make_epub)), "--show", "1"])
    assert result.exit_code == 0
    assert result.stdout == "1. Prologue  (3 words)\n\nPrologue\n\nIt began.\n"


def test_inspect_show_normalizes_text(make_epub: MakeEpub) -> None:
    path = make_epub([Doc("<p>Dr. Lee paid $5 in 1984.</p>", toc_title="One")])
    result = runner.invoke(app, ["inspect", str(path), "--show", "1"])
    assert "Doctor Lee paid five dollars in nineteen eighty-four." in result.stdout


def test_inspect_show_raw_skips_normalization(make_epub: MakeEpub) -> None:
    path = make_epub([Doc("<p>Dr. Lee paid $5 in 1984.</p>", toc_title="One")])
    result = runner.invoke(app, ["inspect", str(path), "--show", "1", "--raw"])
    assert "Dr. Lee paid $5 in 1984." in result.stdout


def test_inspect_show_out_of_range_fails(make_epub: MakeEpub) -> None:
    result = runner.invoke(app, ["inspect", str(two_chapter_book(make_epub)), "--show", "9"])
    assert result.exit_code == 2
    assert "chapter 9 does not exist; the book has 2" in result.stderr


def test_inspect_missing_file_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path / "nope.epub")])
    assert result.exit_code == 1
    assert "error: file not found" in result.stderr


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "inspect" in result.stdout

from typer.testing import CliRunner

from velora.cli import app

runner = CliRunner()


def test_hello_greets_world_by_default() -> None:
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello, world!" in result.stdout


def test_hello_greets_given_name() -> None:
    result = runner.invoke(app, ["hello", "Gian"])
    assert result.exit_code == 0
    assert "Hello, Gian!" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout

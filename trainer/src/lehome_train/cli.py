"""Command-line entry point for the isolated trainer package."""

from __future__ import annotations

import typer


app = typer.Typer(
    help="LeHome GR00T N1.7 trainer.",
    no_args_is_help=True,
)


def _command_group(help_text: str) -> typer.Typer:
    group = typer.Typer(help=help_text, no_args_is_help=True)

    @group.callback()
    def callback() -> None:
        """Expose the command group without loading training dependencies."""

    return group


app.add_typer(_command_group("Dataset inspection and transfer commands."), name="data")
app.add_typer(_command_group("Training environment preparation commands."), name="prepare")
app.add_typer(_command_group("Memorization experiment commands."), name="memorize")
app.add_typer(_command_group("Lightweight training smoke-test commands."), name="smoke")
app.add_typer(_command_group("Full training commands."), name="train")
app.add_typer(_command_group("Training report commands."), name="report")
app.add_typer(_command_group("Artifact synchronization commands."), name="sync")


def main() -> None:
    """Run the trainer CLI."""
    app()


if __name__ == "__main__":
    main()

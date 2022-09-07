"""Console script for crepe_notes."""
import sys
import click
from .crepe_notes import process


@click.command()
@click.argument('f0_path', type=click.Path(exists=True))
@click.argument('audio_path', type=click.Path(exists=True))
def main(f0_path, audio_path, args=None):
    """CREPE notes - get midi or discrete notes from the CREPE pitch tracker"""
    click.echo(click.format_filename(f0_path))
    process(f0_path, audio_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

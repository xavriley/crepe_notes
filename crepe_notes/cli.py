"""Console script for crepe_notes."""
import sys
import click
from .crepe_notes import process


@click.command()
@click.option('--output-label', default='transcription')
@click.option('--sensitivity', type=click.FloatRange(0, 1), default=0.002)
@click.option('--min-duration', type=click.FloatRange(0, 1), default=0.11, help='Minimum duration of a note in seconds')
@click.option('--use-smoothing', is_flag=True, default=False, help='Enable smoothing of confidence')
@click.argument('f0_path', type=click.Path(exists=True))
@click.argument('audio_path', type=click.Path(exists=True))
def main(f0_path, audio_path, output_label, sensitivity, min_duration, use_smoothing):
    """CREPE notes - get midi or discrete notes from the CREPE pitch tracker"""
    click.echo(click.format_filename(f0_path))
    process(f0_path, audio_path, output_label=output_label, sensitivity=sensitivity, use_smoothing=use_smoothing, min_duration=min_duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

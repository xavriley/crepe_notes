"""Console script for crepe_notes."""
import sys
import click
from .crepe_notes import process


@click.command()
@click.option('--output-label', default='transcription')
@click.option('--sensitivity', type=click.FloatRange(0, 1), default=0.001)
@click.option('--min-duration', type=click.FloatRange(0, 1), default=0.03, help='Minimum duration of a note in seconds')
@click.option('--min-velocity', type=click.IntRange(0, 127, clamp=True), default=6, help='Minimum velocity of a note in midi scale (0-127)')
@click.option('--disable-splitting', is_flag=True, default=False, help='Disable detection of repeated notes via onset detection')
@click.option('--tuning-offset', type=click.FloatRange(-100, 100, clamp=True), default=False, help='Manually apply a tuning offset in cents. Fractional numbers are allowed. Set to 0 for no offset, otherwise it will be calculated automatically.')
@click.option('--use-smoothing', is_flag=True, default=False, help='Enable smoothing of confidence')
@click.option('--use-cwd', is_flag=True, default=True, help='If True, write to the cwd of the current command, else write to the parent folder of the f0_path')
@click.argument('f0_path', type=click.Path(exists=True))
@click.argument('audio_path', type=click.Path(exists=True))
@click.help_option()
def main(f0_path, audio_path, output_label, sensitivity, min_duration, min_velocity, disable_splitting, tuning_offset, use_smoothing, use_cwd):
    """CREPE notes - get midi or discrete notes from the CREPE pitch tracker"""
    click.echo(click.format_filename(f0_path))
    process(f0_path, audio_path, output_label=output_label, sensitivity=sensitivity, use_smoothing=use_smoothing,
            min_duration=min_duration, min_velocity=min_velocity, disable_splitting=disable_splitting, use_cwd=use_cwd,
            tuning_offset=tuning_offset)


if __name__ == "__main__":
    main()  # pragma: no cover

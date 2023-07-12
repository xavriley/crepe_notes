"""Console script for crepe_notes."""
import sys
import pathlib
import click
import numpy as np
from .crepe_notes import process, parse_f0, run_crepe

@click.command()
@click.option('--output-label', default='transcription')
@click.option('--sensitivity', type=click.FloatRange(0, 1), default=0.001)
@click.option('--min-duration', type=click.FloatRange(0, 1), default=0.03, help='Minimum duration of a note in seconds')
@click.option('--min-velocity', type=click.IntRange(0, 127, clamp=True), default=6, help='Minimum velocity of a note in midi scale (0-127)')
@click.option('--disable-splitting', is_flag=True, default=False, help='Disable detection of repeated notes via onset detection')
@click.option('--tuning-offset', type=click.FloatRange(-100, 100, clamp=True), default=False, help='Manually apply a tuning offset in cents. Fractional numbers are allowed. Set to 0 for no offset, otherwise it will be calculated automatically.')
@click.option('--use-smoothing', is_flag=True, default=False, help='Enable smoothing of confidence')
@click.option('--use-cwd', is_flag=True, default=False, help='If True, write to the cwd of the current command, else write to the parent folder of the f0_path')
@click.option('--f0', type=click.Path(exists=True))
@click.option('--save-analysis-files', is_flag=True, default=False, help='Save f0, madmom onsets and amp envelope as files')
@click.argument('audio_path', type=click.Path(exists=True, path_type=pathlib.Path))
@click.help_option()
def main(f0, audio_path, output_label, sensitivity, min_duration, min_velocity, disable_splitting, tuning_offset, use_smoothing, use_cwd, save_analysis_files):
    default_f0_path = audio_path.with_suffix('.f0.csv')
    if not default_f0_path.exists() and f0 is None:
        frequency, confidence = run_crepe(audio_path)
    elif f0 is None:
        frequency, confidence = parse_f0(default_f0_path)
    else:
        frequency, confidence = parse_f0(f0)

        
    """CREPE notes - get midi or discrete notes from the CREPE pitch tracker"""
    click.echo(click.format_filename(audio_path))
    process(frequency, confidence, audio_path, output_label=output_label, sensitivity=sensitivity, use_smoothing=use_smoothing,
            min_duration=min_duration, min_velocity=min_velocity, disable_splitting=disable_splitting, use_cwd=use_cwd,
            tuning_offset=tuning_offset, save_analysis_files=save_analysis_files)


if __name__ == "__main__":
    main()  # pragma: no cover

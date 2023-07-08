#!/usr/bin/env python

"""Tests for `crepe_notes` package."""


import unittest
from click.testing import CliRunner

from crepe_notes import crepe_notes
from crepe_notes import cli

import os
import sys
from pathlib import Path

import pretty_midi as pm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import mir_eval
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

class TestCrepe_notes(unittest.TestCase):
    """Tests for `crepe_notes` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def midi_contour_from_f0(self, f0_path):
        freqs, conf = crepe_notes.parse_f0(f0_path)
        tuning_offset = crepe_notes.calculate_tuning_offset(freqs)
        midi_pitch = crepe_notes.freqs_to_midi(freqs) - tuning_offset
        return midi_pitch

    def midi_notes(self, mid_path):
        return pm.PrettyMIDI(str(mid_path)).instruments[0].notes

    def plot_results(self, predicted_mid, gt_mid, f0_path):
        pred = self.midi_notes(predicted_mid)
        gt = self.midi_notes(gt_mid)

        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

        midi_pitch = self.midi_contour_from_f0(f0_path)
        t = np.arange(0, len(midi_pitch)) * 0.01

        plt.plot(t, midi_pitch, '-', color='b')

        latest_note_end = 0
        highest_note = 0
        lowest_note = 127
        for n in pred:
            ax.add_artist(Rectangle((n.start, n.pitch-0.5), n.end - n.start, 1, color='g', alpha=0.5))
            if n.pitch > highest_note:
                highest_note = n.pitch
            if n.pitch < lowest_note:
                lowest_note = n.pitch
            latest_note_end = n.end
            ax.axvline(n.start, alpha=0.5)

        for n in gt:
            ax.add_artist(Rectangle((n.start, n.pitch-0.5), n.end - n.start, 1, color='r', alpha=0.5))
            if n.pitch > highest_note:
                highest_note = n.pitch
            if n.pitch < lowest_note:
                lowest_note = n.pitch
            latest_note_end = n.end

        # need to set axis limits manually
        ax.set_xlim(0, latest_note_end + 0.2)
        ax.set_ylim(lowest_note - 3, highest_note + 3)

        plt.show()
        return

    def calculate_accuracy_metrics(self, predicted_mid, gt_mid):
        ref = pm.PrettyMIDI(str(gt_mid))
        est = pm.PrettyMIDI(str(predicted_mid))

        ref_times = np.array([[n.start, n.end]
                              for n in ref.instruments[0].notes])
        ref_pitches = np.array(
            [pm.note_number_to_hz(n.pitch) for n in ref.instruments[0].notes])

        est_times = np.array([[n.start, n.end]
                              for n in est.instruments[0].notes])
        est_pitches = np.array([
            pm.note_number_to_hz(n.pitch) for n in est.instruments[0].notes
        ])

        return mir_eval.transcription.evaluate(ref_times, ref_pitches,
                                                      est_times, est_pitches)

    def test_step_to_samples(self):
        """Conversion between steps (index of crepe predictions) and samples in audio."""
        assert crepe_notes.steps_to_samples(0, 44100) == 0
        assert crepe_notes.steps_to_samples(1, 44100) == 441
        assert crepe_notes.steps_to_samples(1, 44100, 0.001) == 44
        assert crepe_notes.steps_to_samples(100, 44100) == 44100

    def test_samples_to_steps(self):
        """Conversion between samples index in audio and steps (index of crepe predictions)."""
        assert crepe_notes.samples_to_steps(0, 44100) == 0
        assert crepe_notes.samples_to_steps(1, 44100) == 0
        assert crepe_notes.samples_to_steps(44.1, 44100, 0.001) == 1
        assert crepe_notes.samples_to_steps(44100, 44100) == 100

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help' in help_result.output
        assert 'Show this message and exit.' in help_result.output

    def test_command_line_interface_with_audio_file(self):
        """Test the CLI generates a transcription file"""
        wav_path = Path(TEST_DIR, 'sonny-stitt-lick.wav')
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli.main, ['--min-duration', '0.0001', '--min-velocity', '0', '--disable-splitting', '--use-cwd',
                                              str(wav_path)])
            result_mid_path = Path(os.getcwd(), 'sonny-stitt-lick.transcription.mid')

            assert result_mid_path.exists()
            assert result.exit_code == 0
    
    def test_command_line_interface_with_args(self):
        """Test the CLI generates a transcription file"""
        f0_path = Path(TEST_DIR, 'sonny-stitt-lick.f0.csv')
        wav_path = Path(TEST_DIR, 'sonny-stitt-lick.wav')
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli.main, ['--min-duration', '0.0001', '--min-velocity', '0', '--disable-splitting', '--use-cwd',
                                              '--f0', str(f0_path), str(wav_path)])
            result_mid_path = Path(os.getcwd(), 'sonny-stitt-lick.transcription.mid')

            assert result_mid_path.exists()
            assert result.exit_code == 0

    def test_process(self):
        """Test the main process command"""
        f0_path = Path(TEST_DIR, 'sonny-stitt-lick.f0.csv')
        wav_path = Path(TEST_DIR, 'sonny-stitt-lick.wav')
        gt_transcription = Path(TEST_DIR, 'sonny-stitt-lick.transcription.mid')
        runner = CliRunner()
        with runner.isolated_filesystem():
            result_mid_path = Path(os.getcwd(), 'sonny-stitt-lick.transcription.mid')
            self.assertFalse(result_mid_path.exists())

            freqs, conf = crepe_notes.parse_f0(str(f0_path))

            result = crepe_notes.process(freqs, conf, wav_path, sensitivity=0.001, min_duration=0.0001, tuning_offset=40, min_velocity=0, disable_splitting=True, use_cwd=True)
            assert result_mid_path.exists()

            # self.plot_results(result_mid_path, gt_transcription, f0_path)
            metrics = self.calculate_accuracy_metrics(result_mid_path, gt_transcription)
            score = metrics['F-measure_no_offset']

            print("process: ", score)
            assert score > 0.91

    def test_process_min_duration(self):
        """Test the min duration parameter does remove short notes"""
        f0_path = Path(TEST_DIR, 'sonny-stitt-lick.f0.csv')
        wav_path = Path(TEST_DIR, 'sonny-stitt-lick.wav')
        gt_transcription = Path(TEST_DIR, 'sonny-stitt-lick.transcription.mid')
        runner = CliRunner()
        with runner.isolated_filesystem():
            result_mid_path = Path(os.getcwd(), 'sonny-stitt-lick.transcription.mid')
            self.assertFalse(result_mid_path.exists())

            freqs, conf = crepe_notes.parse_f0(str(f0_path))

            result = crepe_notes.process(freqs, conf, wav_path, min_duration=0.03, sensitivity=0.001, tuning_offset=40, min_velocity=0, disable_splitting=False, use_cwd=True)
            assert result_mid_path.exists()

            # self.plot_results(result_mid_path, gt_transcription, f0_path)

            for note_length in [n.end - n.start for n in self.midi_notes(result_mid_path)]:
                with self.subTest(i=note_length):
                    assert note_length >= 0.03

    def test_process_bass(self):
        """Test on double bass"""
        f0_path = Path(TEST_DIR, 'hymmj-bass-8-bars.f0.csv')
        wav_path = Path(TEST_DIR, 'hymmj-bass-8-bars.wav')
        gt_transcription = Path(TEST_DIR, 'hymmj-bass-8-bars.gt.mid')
        runner = CliRunner()
        with runner.isolated_filesystem():
            result_mid_path = Path(os.getcwd(), 'hymmj-bass-8-bars.transcription.mid')
            self.assertFalse(result_mid_path.exists())

            freqs, conf = crepe_notes.parse_f0(str(f0_path))

            result = crepe_notes.process(freqs, conf, wav_path, min_duration=0.03, sensitivity=0.001, min_velocity=0, disable_splitting=False, use_cwd=True)
            assert result_mid_path.exists()

            print(result_mid_path)
            # self.plot_results(result_mid_path, gt_transcription, f0_path)

            metrics = self.calculate_accuracy_metrics(result_mid_path, gt_transcription)
            score = metrics['F-measure_no_offset']

            print("process bass: ", score)
            assert score > 0.89

            # additional check for invalid midi notes with no note-off
            for note_length in [n.end - n.start for n in self.midi_notes(result_mid_path)]:
                with self.subTest(i=note_length):
                    assert note_length < 1

    def test_process_chad_lb(self):
        """Test on difficult saxophone lick"""
        f0_path = Path(TEST_DIR, 'attya-monster-lick.f0.csv')
        wav_path = Path(TEST_DIR, 'attya-monster-lick.wav')
        gt_transcription = Path(TEST_DIR, 'attya-monster-lick.gt.mid')
        runner = CliRunner()
        with runner.isolated_filesystem():
            result_mid_path = Path(os.getcwd(), 'attya-monster-lick.transcription.mid')
            self.assertFalse(result_mid_path.exists())

            freqs, conf = crepe_notes.parse_f0(str(f0_path))

            result = crepe_notes.process(freqs, conf, wav_path, min_duration=0.03, use_cwd=True)
            assert result_mid_path.exists()

            print(result_mid_path)
            # self.plot_results(result_mid_path, gt_transcription, f0_path)

            metrics = self.calculate_accuracy_metrics(result_mid_path, gt_transcription)
            score = metrics['F-measure_no_offset']

            print("process chad lb: ", score)
            assert score > 0.86
            assert score < 0.87

    def test_process_charlie_parker(self):
        """Test on slurred repeated notes"""
        f0_path = Path(TEST_DIR, 'cp_suede_shoes_repeated_notes.f0.csv')
        wav_path = Path(TEST_DIR, 'cp_suede_shoes_repeated_notes.wav')
        gt_transcription = Path(TEST_DIR, 'cp_suede_shoes_repeated_notes.gt.mid')
        runner = CliRunner()
        with runner.isolated_filesystem():
            result_mid_path = Path(os.getcwd(), 'cp_suede_shoes_repeated_notes.transcription.mid')
            self.assertFalse(result_mid_path.exists())
            
            freqs, conf = crepe_notes.parse_f0(str(f0_path))

            result = crepe_notes.process(freqs, conf, wav_path, min_duration=0.03, use_cwd=True)
            assert result_mid_path.exists()

            # print(result_mid_path)
            # self.plot_results(result_mid_path, gt_transcription, f0_path)

            metrics = self.calculate_accuracy_metrics(result_mid_path, gt_transcription)
            score = metrics['F-measure_no_offset']

            assert score > 0.78 and score < 0.79

    def test_process_slurs(self):
        """Test on slurred notes a semitone apart"""
        f0_path = Path(TEST_DIR, 'attya_filosax_part_4_slurs.f0.csv')
        wav_path = Path(TEST_DIR, 'attya_filosax_part_4_slurs.wav')
        gt_transcription = Path(TEST_DIR, 'attya_filosax_part_4_slurs.gt.mid')
        runner = CliRunner()
        with runner.isolated_filesystem():
            result_mid_path = Path(os.getcwd(), 'attya_filosax_part_4_slurs.transcription.mid')
            self.assertFalse(result_mid_path.exists())
            
            freqs, conf = crepe_notes.parse_f0(str(f0_path))

            result = crepe_notes.process(freqs, conf, wav_path, min_duration=0.03, use_cwd=True)
            assert result_mid_path.exists()

            # print(result_mid_path)
            # self.plot_results(result_mid_path, gt_transcription, f0_path)

            metrics = self.calculate_accuracy_metrics(result_mid_path, gt_transcription)
            score = metrics['F-measure_no_offset']

            assert score > 0.78 and score < 0.79
    
    def test_filosax_full(self):
        """Get results for full Filosax dataset"""

        results = []
        paths = sorted(Path(TEST_DIR, 'Filosax').rglob('Sax.mid'))
        # paths = sorted(Path("/Users/xavriley/Dropbox/PhD/Datasets/Filosax").rglob('Sax.mid'))
        
        for path in paths:
            print(str(path))
            ref = pm.PrettyMIDI(str(path))

            runner = CliRunner()
            with runner.isolated_filesystem():
                result_mid_path = Path(os.getcwd(), path.stem + '.transcription.mid')

                f0_path = Path(path.parent, path.stem + '.f0.csv')
                wav_path = Path(path.parent, path.stem + '.wav')
                
                freqs, conf = crepe_notes.parse_f0(str(f0_path))

                self.assertFalse(result_mid_path.exists())
                result = crepe_notes.process(freqs, conf, wav_path, use_cwd=True)
                assert result_mid_path.exists()

                metrics = self.calculate_accuracy_metrics(str(result_mid_path), str(path))
                results.append(metrics)

        results = pd.DataFrame(results)
        print(results.describe())
        assert(True)

    def test_itm_flute_99_full(self):
        """Get results for full ITM-Flute-99 dataset"""

        results = []
        bp_results = []
        # paths = sorted(Path(TEST_DIR, 'Filosax').rglob('Sax.mid'))
        paths = sorted(Path("/Users/xavriley/Dropbox/PhD/Datasets/GT-ITM-Flute-99").rglob('*.repitched-gt.mid'))
        
        for path in paths:
            print(str(path))
            ref = pm.PrettyMIDI(str(path))

            runner = CliRunner()
            with runner.isolated_filesystem():
                result_mid_path = Path(os.getcwd(), path.stem.replace('izzy_GT_', '').replace('.gt.repitched-gt', '.repitched.rb') + '.transcription.mid')

                f0_path = Path(path.parent, path.stem.replace('izzy_GT_', '').replace('.gt.repitched-gt', '.repitched.rb.f0.csv'))
                wav_path = Path(path.parent, path.stem.replace('izzy_GT_', '').replace('.gt.repitched-gt','.repitched.rb.wav'))
                basic_pitch_path = str(wav_path).replace('.rb.wav', '.rb_basic_pitch.mid')
                
                freqs, conf = crepe_notes.parse_f0(str(f0_path))

                self.assertFalse(result_mid_path.exists())
                result = crepe_notes.process(freqs, conf, wav_path, use_cwd=True, tuning_offset=0.001)
                assert(result_mid_path.exists())

                metrics = self.calculate_accuracy_metrics(str(result_mid_path), str(path))
                bp_metrics = self.calculate_accuracy_metrics(basic_pitch_path, str(path))
                results.append(metrics)
                bp_results.append(bp_metrics)

        if len(results) > 0:
            results = pd.DataFrame(results)
            bp_results = pd.DataFrame(bp_results)
            print(results.describe())
            print(bp_results.describe())
            print("CREPE Notes")
            print(results['Onset_F-measure'].mean())
            print("Basic Pitch")
            print(bp_results['Onset_F-measure'].mean())

            

            




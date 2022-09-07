"""Main module."""
from librosa import load, pitch_tuning, hz_to_midi, time_to_samples
from scipy.signal import find_peaks
import numpy as np
import pretty_midi as pm


def process(f0_path, audio_path):
    y, sr = load(audio_path)
    data = np.genfromtxt(f0_path, delimiter=',', names=True)
    output_filename = f0_path.replace('.f0.csv', '')
    print(output_filename)

    conf = data['confidence']
    freqs = data['frequency']
    amp_envelope = np.abs(y)

    tuning_offset = pitch_tuning(freqs)
    print(f"Tuning offset: {tuning_offset * 100} cents")

    midi_pitch = hz_to_midi(freqs) - tuning_offset
    pitch_changes = np.abs(np.gradient(midi_pitch))

    # TODO: remove magic numbers here
    # clipping and multiplying was mainly to aid plotting
    peaks = find_peaks(np.clip((1 - conf) * pitch_changes, 0, 3) * 25,
                       distance=4,
                       threshold=0.04)[0]

    # TODO: combine the following
    note_list = np.array(
        [round(np.median(midi_pitch[a:b])) for a, b in zip(peaks, peaks[1:])])
    seg_list = np.array(list(zip(note_list, zip(peaks, peaks[1:]))))

    segments = []
    sub_list = []
    for a, b in zip(seg_list, seg_list[1:]):
        # TODO: compute variance in segment to catch glissandi
        # if np.var(midi_pitch[a[1][0]:a[1][1]]) > 1:
        #     continue

        if np.abs(a[0] - b[0]) > 0.5:
            sub_list.append(a)
            segments.append(sub_list)
            sub_list = []
        else:
            sub_list.append(a)

    # catch any segments at the end
    if len(sub_list) > 0:
        segments.append(sub_list)

    output_midi = pm.PrettyMIDI()
    instrument = pm.Instrument(
        program=pm.instrument_name_to_program('Acoustic Grand Piano'))

    velocities = []
    output_notes = []

    for x_s in segments:
        median_y = np.median(np.array([y for y, _ in x_s]))
        seg_start = x_s[0][1][0]
        seg_end = x_s[-1][1][1]
        sample_start = time_to_samples(0.01 * seg_start, sr=sr)
        sample_end = time_to_samples(0.01 * seg_end, sr=sr)
        max_amp = np.max(amp_envelope[sample_start:sample_end])

        # ax.hlines(y=median_y,
        #           xmin=x_s[0][1][0],
        #           xmax=x_s[-1][1][1],
        #           colors='r',
        #           linestyles='dotted')

        velocities.append(max_amp)
        output_notes.append([round(median_y), (x_s[0][1][0], x_s[-1][1][1])])

    velocities = np.array(velocities)
    scaled_velocities = np.interp(velocities, (0, velocities.max()), (0, 127))

    for (n, (note_start, note_end)), v in zip(output_notes, scaled_velocities):
        if v > 3:
            instrument.notes.append(
                pm.Note(start=0.01 * note_start,
                        end=0.01 * note_end,
                        pitch=n,
                        velocity=round(v)))

    print(max(scaled_velocities))
    output_midi.instruments.append(instrument)
    output_midi.write('%s.transcription.mid' % output_filename)

    notes = [y for x_s in segments for y, _ in x_s]

    return True

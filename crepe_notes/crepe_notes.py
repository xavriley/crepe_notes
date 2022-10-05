"""Main module."""
from librosa import load, pitch_tuning, hz_to_midi, time_to_samples
from scipy.signal import find_peaks, hilbert
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
from .one_euro_filter import OneEuroFilter

import os.path


def process(f0_path, audio_path, output_label="transcription", sensitivity=0.002, use_smoothing=False, min_duration=0.04):
    y, sr = load(audio_path)
    data = np.genfromtxt(f0_path, delimiter=',', names=True)
    output_filename = f0_path.replace('.f0.csv', '')
    print(output_filename)
    onsets_path = f0_path.replace('.f0.csv', '.onsets.npz')
    if not os.path.exists(onsets_path):
        print(f"Onsets file not found at {onsets_path}")
        exit()
    onsets_raw = np.load(onsets_path)['activations']
    onsets = np.zeros_like(onsets_raw)
    onsets[find_peaks(onsets_raw, distance=4, height=0.8)[0]] = 1

    conf = np.nan_to_num(data['confidence'])
    t = list(range(0, len(conf)))

    if use_smoothing:
        # The filtered signal
        min_cutoff = 0.002
        beta = 0.7
        smooth_conf = np.zeros_like(conf)
        smooth_conf[0] = conf[0]
        one_euro_filter = OneEuroFilter(
            t[0], conf[0],
            min_cutoff=min_cutoff,
            beta=beta
        )
        for i in range(1, len(t)):
            smooth_conf[i] = one_euro_filter(t[i], conf[i])

    freqs = np.nan_to_num(data['frequency'])
    amp_envelope = np.abs(hilbert(y))

    tuning_offset = pitch_tuning(freqs)
    print(f"Tuning offset: {tuning_offset * 100} cents")

    midi_pitch = np.nan_to_num(hz_to_midi(freqs) - tuning_offset, neginf=0)
    pitch_changes = np.abs(np.gradient(midi_pitch))
    pitch_changes = np.interp(pitch_changes, (pitch_changes.min(), pitch_changes.max()), (0, 1))

    conf_peaks, conf_peak_properties = find_peaks(1-conf,
                                        distance=4,
                                        prominence=sensitivity)
    conf_prominences = conf_peak_properties["prominences"]

    change_point_signal = (1 - conf) * pitch_changes
    change_point_signal = np.interp(change_point_signal, (change_point_signal.min(), change_point_signal.max()), (0, 1))
    peaks, peak_properties = find_peaks(change_point_signal,
                                        distance=4,
                                        prominence=sensitivity)
    prominences = peak_properties["prominences"]

    # scaled_midi_pitch = np.interp(midi_pitch, (0, 127), (0, 1))
    # plt.plot(change_point_signal[0:10000], label='change_points')
    # plt.plot(smooth_conf[0:10000], label='conf')
    # plt.plot(scaled_midi_pitch[0:10000], ',', label='pitch_changes')
    # plt.plot(peaks[peaks < 10000], change_point_signal[peaks[peaks < 10000]], '.')
    # for idx, p in enumerate(conf_prominences[0:500]):
    #     plt.vlines(conf_peaks[idx], 0, p)
    # plt.show()

    # take the amplitudes within 6 sigma of the mean
    # helps to clean up outliers in amplitude scaling as we are not looking for 100% accuracy
    amp_mean = np.mean(amp_envelope)
    amp_sd = np.std(amp_envelope)
    filtered_amp_envelope = [x for x in amp_envelope if (x < amp_mean + 6 * amp_sd)]
    global_max_amp = max(filtered_amp_envelope)
    # print(f"max_amp: {max(amp_envelope)} filtered_max_amp: {global_max_amp}")

    min_scaled_velocity = 15
    min_median_confidence = 1

    segment_list = []
    for a, b in zip(peaks, peaks[1:]):
        a_samp = int(a * (sr * 0.01))
        b_samp = int(b * (sr * 0.01))
        max_amp = np.max(amp_envelope[a_samp:b_samp])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        segment_list.append({
            'pitch': np.round(np.median(midi_pitch[a:b])),
            'conf': np.median(conf[a:b]),
            'transition_strength': 1-conf[a],
            'amplitude': scaled_max_amp,
            'start_idx': a,
            'finish_idx': b,
        })

    notes = []
    sub_list = []
    for a, b in zip(segment_list, segment_list[1:]):
        # TODO: compute variance in segment to catch glissandi
        # if np.var(midi_pitch[a[1][0]:a[1][1]]) > 1:
        #     continue

        if np.abs(a['pitch'] - b['pitch']) > 0.5: # or a['transition_strength'] > 0.4:
            sub_list.append(a)
            notes.append(sub_list)
            sub_list = []
        else:
            sub_list.append(a)

    # catch any segments at the end
    if len(sub_list) > 0:
        notes.append(sub_list)

    output_midi = pm.PrettyMIDI()
    instrument = pm.Instrument(
        program=pm.instrument_name_to_program('Acoustic Grand Piano'))

    velocities = []
    durations = []
    output_notes = []

    for x_s in notes:
        x_s_filt = [x for x in x_s if x['amplitude'] > min_scaled_velocity]
        if len(x_s_filt) == 0:
            continue
        median_pitch = np.median(np.array([y['pitch'] for y in x_s_filt]))
        median_confidence = np.median(np.array([y['conf'] for y in x_s_filt]))
        seg_start = x_s_filt[0]['start_idx']
        seg_end = x_s_filt[-1]['finish_idx']
        time_start = 0.01 * seg_start
        time_end = 0.01 * seg_end
        sample_start = time_to_samples(time_start, sr=sr)
        sample_end = time_to_samples(time_end, sr=sr)
        max_amp = np.max(amp_envelope[sample_start:sample_end])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        valid_amplitude = True # scaled_max_amp > min_scaled_velocity
        valid_confidence = True # median_confidence > 0.1
        valid_duration = True # (time_end - time_start) > min_duration

        if valid_amplitude and valid_confidence and valid_duration:
            output_notes.append({
                'pitch': int(np.round(median_pitch)),
                'velocity': round(scaled_max_amp),
                'start_idx': seg_start,
                'finish_idx': seg_end,
                'conf': median_confidence,
                'transition_strength': x_s[-1]['transition_strength']
            })

    onset_separated_notes = []
    for n in output_notes:
        n_s = n['start_idx']
        n_f = n['finish_idx']

        # if seg_s > 2000 and seg_f < 2700:
        #     plt.plot(midi_pitch[n[0]['start_idx']:n[-1]['finish_idx']])
        #     plt.vlines(seg_s - n[0]['start_idx'], 50, 70, 'r')
        #     plt.vlines(seg_f - seg_s, 50, 70, 'r')
        #     plt.plot(onsets[seg_s:seg_f] * 5 + 50)
        #     plt.show()

        last_onset = 0
        if np.any(onsets[n_s:n_f] > 0.95):
            onset_idxs_within_note = np.argwhere(onsets[n_s:n_f] > 0.95)
            for idx in onset_idxs_within_note:
                if idx[0] > last_onset + int(min_duration / 0.01):
                    new_note = n.copy()
                    new_note['start_idx'] = n_s + last_onset
                    new_note['finish_idx'] = n_s + idx[0]
                    onset_separated_notes.append(new_note)
                    last_onset = idx[0]

        # If there are no valid onsets within the range
        # the following should append a copy of the original note,
        # but if there were splits at onsets then it will also clean up any tails
        # left in the sequence
        new_note = n.copy()
        new_note['start_idx'] = n_s + last_onset
        new_note['finish_idx'] = n_f
        onset_separated_notes.append(new_note)

    timed_output_notes = []
    for n in onset_separated_notes:
        timed_note = n.copy()
        timed_note['start'] = timed_note['start_idx'] * 0.01
        timed_note['finish'] = timed_note['finish_idx'] * 0.01
        timed_output_notes.append(timed_note)

    # import plotext as plttxt
    # print(f"max pitch: {np.unique([n['pitch'] for n in output_notes])}")
    # scaled_velocities = [n['velocity'] for n in output_notes]
    # print(max(scaled_velocities))
    # plttxt.hist(scaled_velocities, 20, label='Velocities')
    # plttxt.title("Velocity distribution")
    # plttxt.show()

    # confs = [n['transition_strength'] for n in segment_list]
    # plt.hist(confs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], label='Transition strengths')
    # plt.title("Transitions")
    # plt.show()
    #
    # durations = [n['finish'] - n['start'] for n in output_notes]
    # plt.hist(durations, 30, label='Durations')
    # plt.title('Durations')
    # plt.show()

    for n in timed_output_notes:
        instrument.notes.append(
            pm.Note(start=n['start'],
                    end=n['finish'],
                    pitch=n['pitch'],
                    velocity=n['velocity']))

    output_midi.instruments.append(instrument)
    output_midi.write(f'{output_filename}.{output_label}.mid')

    return True

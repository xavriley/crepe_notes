"""Main module."""
from librosa import load, pitch_tuning, hz_to_midi, time_to_samples
from scipy.signal import find_peaks, hilbert
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
from .one_euro_filter import OneEuroFilter


def process(f0_path, audio_path, output_label="transcription", sensitivity=0.002, use_smoothing=False, min_duration=0.11):
    y, sr = load(audio_path)
    data = np.genfromtxt(f0_path, delimiter=',', names=True)
    output_filename = f0_path.replace('.f0.csv', '')
    print(output_filename)

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

    segment_list = []
    for a, b in zip(peaks, peaks[1:]):
        segment_list.append({
            'pitch': np.round(np.median(midi_pitch[a:b])),
            'conf': np.median(conf[a:b]),
            'transition_strength': 1-conf[a],
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

    min_scaled_velocity = 15
    min_median_confidence = 1

    # take the amplitudes within 6 sigma of the mean
    # helps to clean up outliers in amplitude scaling as we are not looking for 100% accuracy
    amp_mean = np.mean(amp_envelope)
    amp_sd = np.std(amp_envelope)
    filtered_amp_envelope = [x for x in amp_envelope if (x < amp_mean + 6 * amp_sd)]
    global_max_amp = max(filtered_amp_envelope)
    # print(f"max_amp: {max(amp_envelope)} filtered_max_amp: {global_max_amp}")

    for x_s in notes:
        median_pitch = np.median(np.array([y['pitch'] for y in x_s]))
        median_confidence = np.median(np.array([y['conf'] for y in x_s]))
        seg_start = x_s[0]['start_idx']
        seg_end = x_s[-1]['finish_idx']
        time_start = 0.01 * seg_start
        time_end = 0.01 * seg_end
        sample_start = time_to_samples(time_start, sr=sr)
        sample_end = time_to_samples(time_end, sr=sr)
        max_amp = np.max(amp_envelope[sample_start:sample_end])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        valid_amplitude = scaled_max_amp > min_scaled_velocity
        valid_confidence = median_confidence > 0.1
        valid_duration = (time_end - time_start) > min_duration

        if valid_amplitude and valid_confidence and valid_duration:
            output_notes.append({
                'pitch': int(np.round(median_pitch)),
                'velocity': round(scaled_max_amp),
                'start': time_start,
                'finish': time_end,
                'conf': median_confidence,
                'transition_strength': x_s[-1]['transition_strength']
            })

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

    for n in output_notes:
        instrument.notes.append(
            pm.Note(start=n['start'],
                    end=n['finish'],
                    pitch=n['pitch'],
                    velocity=n['velocity']))

    output_midi.instruments.append(instrument)
    output_midi.write(f'{output_filename}.{output_label}.mid')

    return True

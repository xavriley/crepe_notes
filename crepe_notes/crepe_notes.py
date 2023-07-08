"""Main module."""
from librosa import load, get_samplerate, pitch_tuning, hz_to_midi, time_to_samples, onset, stft
from scipy.signal import find_peaks, hilbert, peak_widths, butter, filtfilt, resample
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
from .one_euro_filter import OneEuroFilter

import os.path
from pathlib import Path


def steps_to_samples(step_val, sr, step_size=0.01):
    return int(step_val * (sr * step_size))


def samples_to_steps(sample_val, sr, step_size=0.01):
    return int(sample_val / (sr * step_size))


def freqs_to_midi(freqs, tuning_offset=0):
    return np.nan_to_num(hz_to_midi(freqs) - tuning_offset, neginf=0)


def calculate_tuning_offset(freqs):
    tuning_offset = pitch_tuning(freqs)
    print(f"Tuning offset: {tuning_offset * 100} cents")
    return tuning_offset


def parse_f0(f0_path):
    data = np.genfromtxt(f0_path, delimiter=',', names=True)
    return np.nan_to_num(data['frequency']), np.nan_to_num(data['confidence'])


def process(freqs,
            conf,
            audio_path,
            output_label="transcription",
            sensitivity=0.001,
            use_smoothing=False,
            min_duration=0.03,
            min_velocity=6,
            disable_splitting=False,
            use_cwd=True,
            tuning_offset=False,
            detect_amplitude=True):
    
    cached_amp_envelope_path = audio_path.with_suffix(".amp_envelope.npz")
    if cached_amp_envelope_path.exists():
        # if we have a cached amplitude envelope, no need to load audio
        filtered_amp_envelope = np.load(cached_amp_envelope_path, allow_pickle=True)['filtered_amp_envelope']
        # sr = get_samplerate(audio_path)
        sr = 44100 # TODO: this is just to make tests work and could lead to confusion
    else:
        try:
            y, sr = load(str(audio_path), sr=None)
        except:
            print("Error loading audio file. Amplitudes will be set to 80")
            detect_amplitude = False
            pass

        amp_envelope = np.abs(hilbert(y))

        scaled_amp_envelope = np.interp(amp_envelope, (amp_envelope.min(), amp_envelope.max()), (0, 1))
        # low pass filter the amplitude envelope
        b, a = butter(4, 50, 'low', fs=sr)
        filtered_amp_envelope = filtfilt(b, a, scaled_amp_envelope)[::(sr//100)]

        np.savez(cached_amp_envelope_path, filtered_amp_envelope=filtered_amp_envelope)
    
    if use_cwd:
        # write to location that the bin was run from
        output_filename = audio_path.stem
    else:
        # write to same folder as the orignal audio file
        output_filename = str(audio_path.parent) + "/" + audio_path.stem

    print(os.path.abspath(audio_path))

    if not disable_splitting:
        onsets_path = str(audio_path.with_suffix('.onsets.npz'))
        if not os.path.exists(onsets_path):
            print(f"Onsets file not found at {onsets_path}")
            exit()
        onsets_raw = np.load(onsets_path, allow_pickle=True)['activations']
        onsets = np.zeros_like(onsets_raw)
        onsets[find_peaks(onsets_raw, distance=4, height=0.8)[0]] = 1

    t = list(range(0, len(conf)))

    if use_smoothing:
        # The filtered signal
        min_cutoff = 0.002
        beta = 0.7
        smooth_conf = np.zeros_like(conf)
        smooth_conf[0] = conf[0]
        one_euro_filter = OneEuroFilter(t[0],
                                        conf[0],
                                        min_cutoff=min_cutoff,
                                        beta=beta)
        for i in range(1, len(t)):
            smooth_conf[i] = one_euro_filter(t[i], conf[i])

    if tuning_offset == False:
        tuning_offset = calculate_tuning_offset(freqs)
    else:
        tuning_offset = tuning_offset / 100

    midi_pitch = freqs_to_midi(freqs, tuning_offset)
    pitch_changes = np.abs(np.gradient(midi_pitch))
    pitch_changes = np.interp(pitch_changes,
                              (pitch_changes.min(), pitch_changes.max()),
                              (0, 1))

    conf_peaks, conf_peak_properties = find_peaks(1 - conf,
                                                  distance=4,
                                                  prominence=sensitivity)
    conf_prominences = conf_peak_properties["prominences"]

    change_point_signal = (1 - conf) * pitch_changes
    change_point_signal = np.interp(
        change_point_signal,
        (change_point_signal.min(), change_point_signal.max()), (0, 1))
    peaks, peak_properties = find_peaks(change_point_signal,
                                        distance=4,
                                        prominence=sensitivity)
    _, _, transition_starts, transition_ends = peak_widths(change_point_signal, peaks, rel_height=0.5)
    transition_starts = list(map(int, np.round(transition_starts)))
    transition_ends = list(map(int, np.round(transition_ends)))

    transitions = [(s, f, 'transition') for (s, f) in zip(transition_starts, transition_ends)]
    note_starts = [0] + transition_ends
    note_ends = transition_starts + [len(change_point_signal) + 1]
    note_regions = [(s, f, 'note') for (s, f) in (zip(note_starts, note_ends))]

    show_plots = False
    if show_plots:
        fig, axs = plt.subplots(4, 1, sharex=True)
        axs[0].plot(midi_pitch)
        # [axs[0].fill_between(x, 56, 75, alpha=0.5) for x in note_regions]
        axs[1].plot(change_point_signal)
        axs[1].plot(peaks, change_point_signal[peaks], 'x')
        
        
        spectral_flux = onset.onset_strength(y=y,
            sr=sr, 
            hop_length=(sr // 100),
            center=False,
            )
        spectral_flux = np.interp(spectral_flux, (spectral_flux.min(), spectral_flux.max()), (0, 1))
        inv_conf = np.square(1 - conf)

        axs[2].plot(inv_conf * spectral_flux)
        axs[2].plot(spectral_flux)
        
        if detect_amplitude:
            filtered_amp_envelope_for_plot = np.interp(filtered_amp_envelope, (filtered_amp_envelope.min(), filtered_amp_envelope.max()), (0, 1))

            axs[2].plot(filtered_amp_envelope_for_plot)
            
            amp_onsets = np.gradient(filtered_amp_envelope)
            amp_onsets = np.interp(amp_onsets, (amp_onsets.min(), amp_onsets.max()), (-1, 1))
            # get positive values only
            amp_offsets = np.abs(np.clip(amp_onsets, -1, 0))
            amp_onsets = np.clip(amp_onsets, 0, 1)

            axs[2].plot(amp_offsets)
            axs[2].plot(amp_onsets)

        # one idea is that a slurred note has a lowering in
        # amplitude together with a peak in the spectral flux
        # thing = np.zeros_like(spectral_flux)
        # thing[spectral_flux > amp_offsets] = 1
        
        # thing_regions = np.nonzero(thing)
        
        # TODO: HERE
        # another way to think about this is to combine
        # periods of volatility in the spectral flux with
        # downward trends in the amplitude envelope
        
        # axs[3].plot(thing)
        
        axs[3].plot(spectral_flux)
        axs[3].plot(1-conf)
        axs[3].plot(filtered_amp_envelope)
        axs[3].legend(['spectral_flux', '1-conf', 'filtered_amp_envelope'])


        # add legend
        axs[0].legend(['midi_pitch'])
        axs[1].legend(['change_point_signal', 'peaks'])
        axs[2].legend(['conf*spectral_flux', 'spectral_flux', 'filtered_amp_envelope', 'amp_offsets', 'amp_onsets'])

        plt.show()

    prominences = peak_properties["prominences"]

    # scaled_midi_pitch = np.interp(midi_pitch, (0, 127), (0, 1))
    # plt.plot(change_point_signal[0:10000], label='change_points')
    # plt.plot(smooth_conf[0:10000], label='conf')
    # plt.plot(scaled_midi_pitch[0:10000], ',', label='pitch_changes')
    # plt.plot(peaks[peaks < 10000], change_point_signal[peaks[peaks < 10000]], '.')
    # for idx, p in enumerate(conf_prominences[0:500]):
    #     plt.vlines(conf_peaks[idx], 0, p)
    # plt.show()

    if detect_amplitude:
        # take the amplitudes within 6 sigma of the mean
        # helps to clean up outliers in amplitude scaling as we are not looking for 100% accuracy
        amp_mean = np.mean(filtered_amp_envelope)
        amp_sd = np.std(filtered_amp_envelope)
        # filtered_amp_envelope = amp_envelope.copy()
        filtered_amp_envelope[filtered_amp_envelope > amp_mean + (6 * amp_sd)] = 0
        global_max_amp = max(filtered_amp_envelope)
        # print(f"max_amp: {max(amp_envelope)} filtered_max_amp: {global_max_amp}")

    min_median_confidence = 1

    segment_list = []
    # for a, b in zip(peaks, peaks[1:]):
    for a, b, label in sum(zip(note_regions, transitions), ()):
        if label == 'transition':
            continue

        # Handle an edge case where rounding could cause
        # an end index for a note to be before the start index
        if a > b:
            continue
        elif b - a <= 1:
            continue

        a_samp = steps_to_samples(a, sr)
        b_samp = steps_to_samples(b, sr)

        if detect_amplitude:
            max_amp = np.max(filtered_amp_envelope[a:b])
            scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))
        else:
            scaled_max_amp = 80

        segment_list.append({
            'pitch': np.round(np.median(midi_pitch[a:b])),
            'conf': np.median(conf[a:b]),
            'transition_strength': 1 - conf[a], # this no longer makes sense
            'amplitude': scaled_max_amp,
            'start_idx': a,
            'finish_idx': b,
        })

    # MERGE SEGMENTS WITH SAME MEDIAN PITCH
    notes = []
    sub_list = []
    for a, b in zip(segment_list, segment_list[1:]):
        # TODO: compute variance in segment to catch glissandi
        # if np.var(midi_pitch[a[1][0]:a[1][1]]) > 1:
        #     continue

        if np.abs(a['pitch'] -
                  b['pitch']) > 0.5:  # or a['transition_strength'] > 0.4:
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

    # FILTER MIN VELOCITY AND MIN DURATION
    for x_s in notes:
        x_s_filt = [x for x in x_s if x['amplitude'] > min_velocity]
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
        max_amp = np.max(filtered_amp_envelope[seg_start:seg_end])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        valid_amplitude = True  # scaled_max_amp > min_velocity
        valid_confidence = True  # median_confidence > 0.1
        valid_duration = True  # (time_end - time_start) > min_duration

        if valid_amplitude and valid_confidence and valid_duration:
            output_notes.append({
                'pitch':
                    int(np.round(median_pitch)),
                'velocity':
                    round(scaled_max_amp),
                'start_idx':
                    seg_start,
                'finish_idx':
                    seg_end,
                'conf':
                    median_confidence,
                'transition_strength':
                    x_s[-1]['transition_strength']
            })

    # RE-SEPARATE REPEATED NOTES USING ONSET DETECTION
    if not disable_splitting:
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
            if np.any(onsets[n_s:n_f] > 0.7):
                onset_idxs_within_note = np.argwhere(onsets[n_s:n_f] > 0.7)
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
            output_notes = onset_separated_notes

    if detect_amplitude:
        # AMPLITUDE TRIMMING
        timed_output_notes = []
        for n in output_notes:
            timed_note = n.copy()

            # Adjusting the start time to meet a minimum amp threshold
            s = timed_note['start_idx']
            f = timed_note['finish_idx']

            if f - s > (min_duration / 0.01):
                noise_floor = 0.01  # this will vary depending on the signal
                s_samp = steps_to_samples(s, sr)
                f_samp = steps_to_samples(f, sr)
                s_adj_samp_all = s_samp + np.where(
                    filtered_amp_envelope[s:f] > noise_floor)[0]

                if len(s_adj_samp_all) > 0:
                    s_adj_samp_idx = s_adj_samp_all[0]
                else:
                    continue

                s_adj = samples_to_steps(s_adj_samp_idx, sr)

                f_adj_samp_idx = f_samp - np.where(
                    np.flip(filtered_amp_envelope[s:f]) > noise_floor)[0][0]
                if f_adj_samp_idx > f_samp or f_adj_samp_idx < 1:
                    print("something has gone wrong")

                f_adj = samples_to_steps(f_adj_samp_idx, sr)
                if f_adj > f or f_adj < 1:
                    print("something has gone more wrong")

                timed_note['start'] = s_adj * 0.01
                timed_note['finish'] = f_adj * 0.01
                timed_output_notes.append(timed_note)
            else:
                timed_note['start'] = s * 0.01
                timed_note['finish'] = f * 0.01

    # s = 1400
    # f = 1425
    # plt.plot(amp_envelope[steps_to_samples(s, sr):steps_to_samples(f, sr)])
    # plt.hlines(1, timed_output_notes[0]['start_idx'] - s, f, alpha=0.5)
    # plt.plot()

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
        if n['start'] >= n['finish']:
            continue

        instrument.notes.append(
            pm.Note(start=n['start'],
                    end=n['finish'],
                    pitch=n['pitch'],
                    velocity=n['velocity']))

    output_midi.instruments.append(instrument)
    output_midi.write(f'{output_filename}.{output_label}.mid')

    return f"{output_filename}.{output_label}.mid"

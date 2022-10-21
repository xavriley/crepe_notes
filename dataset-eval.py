import mir_eval
from pathlib import Path
import pretty_midi as pm
import numpy as np
import pandas as pd


def evaluate(midi_path='*.crepe_notes.mid',
             midi_replace_str='.crepe_notes.mid',
             output_label='crepe_notes'):
    results = []

    merge_tracks = True

    paths = sorted(Path('./').rglob(midi_path))
    for path in paths:
        print(f"{output_label}: {path}")
        est_path = str(path)
        ref_path = str(path).replace(midi_replace_str, '.gt.mid')

        ref = pm.PrettyMIDI(ref_path)
        est = pm.PrettyMIDI(est_path)

        ref_times = np.array([[n.start, n.end]
                              for n in ref.instruments[0].notes])
        ref_pitches = np.array(
            [pm.note_number_to_hz(n.pitch) for n in ref.instruments[0].notes])

        if merge_tracks:
            est_times = np.array(
                [[n.start, n.end]
                 for inst_notes in map(lambda i: i.notes, est.instruments)
                 for n in inst_notes])
            est_pitches = np.array([
                pm.note_number_to_hz(n.pitch)
                for inst_notes in map(lambda i: i.notes, est.instruments)
                for n in inst_notes
            ])
        else:
            est_times = np.array([[n.start, n.end]
                                  for n in est.instruments[0].notes])
            est_pitches = np.array([
                pm.note_number_to_hz(n.pitch) for n in est.instruments[0].notes
            ])

        first_ref_note = np.min(ref_times)
        last_ref_note = np.max(ref_times)
        est_times_valid_idxs = np.unique(
            np.where((est_times > first_ref_note)
                     & (est_times < last_ref_note))[0])

        est_times = est_times[est_times_valid_idxs]
        est_pitches = est_pitches[est_times_valid_idxs]

        eval_result = mir_eval.transcription.evaluate(ref_times, ref_pitches,
                                                      est_times, est_pitches)
        eval_result['file'] = str(path)
        results.append(eval_result)
    df = pd.DataFrame(results)
    df.to_pickle(f'{output_label}_eval.pkl')
    print(df.describe())


# evaluate('Sax.mt3.mid', '.mt3.mid', 'mt3')
# evaluate('Sax.crepe_notes_amp_trimming.mid', '.crepe_notes_amp_trimming.mid', 'crepe_notes_amp_trimming')
# evaluate('Sax_vamp_pyin_pyin_notes.mid', '_vamp_pyin_pyin_notes.mid', 'pyin_notes')
# evaluate('Sax.crepe_notes_with_onsets.mid', '.crepe_notes_with_onsets.mid', 'crepe_notes_with_onsets')

# evaluate('*_basic_pitch.mid', '_basic_pitch.mid', 'basic_pitch')
# evaluate('*.mt3.mid', '.mt3.mid', 'mt3')
# evaluate('*.cn_25ms_min.mid', '.cn_25ms_min.mid', 'crepe_notes-min-dur-25ms')
# evaluate('*.crepe_notes-min-dur-11ms.mid', '.crepe_notes-min-dur-11ms.mid', 'crepe_notes-min-dur-11ms')
# evaluate('*.crepe_notes-min-dur-11ms-no-tuning.mid',
# evaluate('*.crepe_notes-min-dur-25ms-no-tuning.mid',
#          '.crepe_notes-min-dur-25ms-no-tuning.mid',
#          'crepe_notes-min-dur-25ms-no-tuning')
# evaluate('*_vamp_pyin_pyin_notes.mid', '_vamp_pyin_pyin_notes.mid', 'pyin_notes')
evaluate('*.cn_25ms_min_tiny.mid', '.cn_25ms_min_tiny.mid', 'crepe_notes-min-dur-25ms-tiny')


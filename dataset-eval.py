import mir_eval
from pathlib import Path
import pretty_midi as pm
import numpy as np
import pandas as pd

def evaluate(midi_path='*.crepe_notes.mid', midi_replace_str='.crepe_notes.mid', output_label='crepe_notes'):
    results = []

    paths = sorted(Path('./').rglob(midi_path))
    for path in paths:
        print(f"{output_label}: {path}")
        est_path = str(path)
        ref_path = str(path).replace(midi_replace_str, '.mid')

        ref = pm.PrettyMIDI(ref_path)
        est = pm.PrettyMIDI(est_path)

        ref_times = np.array([[n.start, n.end] for n in ref.instruments[0].notes])
        est_times = np.array([[n.start, n.end] for n in est.instruments[0].notes])

        ref_pitches = np.array(
            [pm.note_number_to_hz(n.pitch) for n in ref.instruments[0].notes])
        est_pitches = np.array(
            [pm.note_number_to_hz(n.pitch) for n in est.instruments[0].notes])

        results.append(
            mir_eval.transcription.evaluate(ref_times, ref_pitches, est_times,
                                            est_pitches))
    df = pd.DataFrame(results)
    df.to_pickle(f'{output_label}_eval.pkl')
    print(df.describe())

# evaluate()
# evaluate('Sax_vamp_pyin_pyin_notes.mid', '_vamp_pyin_pyin_notes.mid', 'pyin_notes')
# evaluate('Sax_basic_pitch.mid', '_basic_pitch.mid', 'basic_pitch')
evaluate('Sax.crepe_notes_with_onsets.mid', '.crepe_notes_with_onsets.mid', 'crepe_notes_with_onsets')

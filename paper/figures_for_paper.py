from matplotlib import pyplot as plt
import numpy as np
import pretty_midi as pm
from librosa import hz_to_midi
from librosa import time_to_frames
from scipy.signal import find_peaks

sensitivity=0.002
f0_path = '/Users/xavriley/Dropbox/PhD/Datasets/Filosax/Participant 4/17/Sax.f0.csv'
gt_midi = pm.PrettyMIDI('/Users/xavriley/Dropbox/PhD/Datasets/Filosax/Participant 4/17/Sax.mid')
data = np.genfromtxt(f0_path, delimiter=',', names=True)

freqs = np.nan_to_num(data['frequency'])
conf = np.nan_to_num(data['confidence'])
tuning_offset = 0

midi_pitch = np.nan_to_num(hz_to_midi(freqs) - tuning_offset, neginf=0)
pitch_changes = np.abs(np.gradient(midi_pitch))
pitch_changes = np.interp(pitch_changes, (pitch_changes.min(), pitch_changes.max()), (0, 1))

change_point_signal = (1 - conf) * pitch_changes
change_point_signal = np.interp(change_point_signal, (change_point_signal.min(), change_point_signal.max()), (0, 1))
peaks, peak_properties = find_peaks(change_point_signal,
                                    distance=4,
                                    prominence=sensitivity)
prominences = peak_properties["prominences"]

s = int(288.98 * 100)
f = int(290.41 * 100)
extract = [n for n in gt_midi.instruments[0].notes if n.start > s/100-0.01 and n.end < f/100+0.01]
extract_notes = np.array([n.pitch for n in extract])

t = np.linspace(s/100, f/100, f-s)

plt.figure(figsize=(3.5, 2))
plt.grid()
plt.plot(t, midi_pitch[s:f], '.', markeredgecolor='k', markerfacecolor='w', zorder=10)
for n in extract:
    plt.hlines(n.pitch, n.start, n.end, color='#888888', lw=10, zorder=1)
plt.ylim(extract_notes.min() - 3, extract_notes.max() + 3)
# ax[0].set_title('a)', loc='left')
plt.xlabel('Time (s)')
plt.ylabel('MIDI Pitch')

plt.tight_layout()
plt.savefig('icassp_figure-a.eps')
plt.show()

plt.figure(figsize=(3.5, 2))
plt.grid()
plt.plot(t, conf[s:f])
conf_peaks = find_peaks(1-conf[s:f], distance=4, prominence=0.002)[0]
plt.plot(t[conf_peaks], conf[s:f][conf_peaks], '.', markeredgecolor='b', markerfacecolor='w')
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig(f'icassp_figure-b.eps')
plt.show()

plt.figure(figsize=(3.5, 2))
plt.grid()
plt.plot(t, pitch_changes[s:f])
peaks = find_peaks(pitch_changes[s:f], distance=4, prominence=0.002)[0]
plt.plot(t[peaks], pitch_changes[s:f][peaks], '.', markeredgecolor='b', markerfacecolor='w')

plt.ylim(0, 1)
plt.ylabel('Abs. Pitch Gradient')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig(f'icassp_figure-c.eps', bbox_inches='tight')
plt.show()

plt.figure(figsize=(3.5, 2))
plt.grid()
plt.ylim(0, 0.5)
plt.plot(t, change_point_signal[s:f])
peaks = find_peaks(change_point_signal[s:f], distance=4, prominence=0.002)[0]
plt.plot(t[peaks], change_point_signal[s:f][peaks], '.', markeredgecolor='b', markerfacecolor='w')

plt.ylabel('1-Conf * Pitch Grad')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig('icassp_figure-d.eps', bbox_inches='tight')
plt.show()

import librosa
import numpy as np
from scipy.signal import find_peaks

f0, confidence = np.loadtxt('crepe_output.csv')

midi_grad = np.gradient(librosa.hz_to_midi(f0))
norm_midi_gradient = np.interp(midi_grad,
                            (midi_grad.min(), midi_grad.max()),
                            (0, 1))
combined_signal = (1-conf) * norm_midi_grad
segmentation_idxs = find_peaks(combined_signal,
                               distance=4,
                               prominence=0.002)

"""Dataset e utility audio/STFT.

CODICE DELL'ALTRO TESISTA — copiato VERBATIM dalla cella 23 del notebook
originale. Non modificato: eventuali problemi sono documentati in REVIEW.md.
(Solo l'header di import è stato reso esplicito per usarlo come modulo.)
"""

from tqdm import tqdm
import pickle
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, lfilter, freqz

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parametri globali
sample_rate = 22050  # sample rate degli audio
cutoff = 7000  # frequenza di cutoff
max_len_wave = 661560
stride = 1024
window_size = 4096

def butter_lowpass(cutoff, fs, order=2):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    #print(y.size())
    return y

def remove_window(audio_windowed):
    n_frames = audio_windowed.shape[0]
    output_len = (n_frames - 1) * stride + window_size

    out = np.zeros(661500)

    out[0:window_size] = audio_windowed[0]

    for i in range(1, n_frames):
        start = i * stride
        end = min(start + window_size, 661500)
        window_len = end - start

        if window_len > 0:
            overlap_start = max(0, start)
            overlap_end = min(start + stride, 661500)

            if overlap_end > overlap_start:
                out[overlap_start:overlap_end] = (out[overlap_start:overlap_end] +
                                                   audio_windowed[i][:overlap_end-overlap_start]) / 2
            if overlap_end < end:
                out[overlap_end:end] = audio_windowed[i][overlap_end-start:window_len]

    return out

def frame_audio(audio, frame_len=4096, hop=1024):
    n_frames = 1 + (len(audio) - frame_len) // hop
    frames = np.zeros((n_frames, frame_len), dtype=audio.dtype)
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        frames[i] = audio[start:end]
    return frames


# ============== DATASET ==============

class AudioSTFTDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, stft_paths):
        """
        Dataset che gestisce correttamente audio e STFT.

        Args:
            audio_paths (list): Lista dei path delle tracce audio (.npy files).
            stft_paths (list): Lista dei path delle STFT corrispondenti (.npy files).

        Returns:
            audio_filtered_window: Audio filtrato e diviso in finestre [n_frames, window_size]
            stft_filtered: STFT dell'audio filtrato
            stft_gt: STFT ground truth (originale)
            audio_gt: Audio ground truth (originale, non windowed)
        """
        self.audio_paths = audio_paths
        self.stft_paths = stft_paths

        self.valid_indices = []
        for i, (audio_path, stft_path) in enumerate(zip(audio_paths, stft_paths)):
            if os.path.exists(audio_path) and os.path.exists(stft_path):
                self.valid_indices.append(i)

        if len(self.valid_indices) == 0:
            raise ValueError("Nessun file valido trovato nei path specificati!")

        print(f"Dataset inizializzato con {len(self.valid_indices)} file validi su {len(audio_paths)} totali")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        try:
            raw_audio = np.load(self.audio_paths[real_idx])
        except Exception as e:
            print(f"Errore nel caricare {self.audio_paths[real_idx]}: {e}")
            return (torch.zeros(1, window_size),
                    torch.zeros(2049, 1, dtype=torch.complex64),
                    torch.zeros(2049, 1, dtype=torch.complex64),
                    np.zeros(661500))

        if len(raw_audio) > 661500:
            raw_audio = raw_audio[:661500]
        elif len(raw_audio) < 661500:
            raw_audio = np.pad(raw_audio, (0, 661500 - len(raw_audio)), mode='constant')

        filtered_audio = butter_lowpass_filter(raw_audio, cutoff, sample_rate, order=2)

        audio_windowed = frame_audio(filtered_audio, frame_len=window_size, hop=stride)

        try:
            stft_gt = np.load(self.stft_paths[real_idx])
        except Exception as e:
            print(f"Errore nel caricare STFT {self.stft_paths[real_idx]}: {e}")
            stft_gt = librosa.stft(
                raw_audio,
                n_fft=4096,
                win_length=4096,
                window='hann'
            )

        stft_train = librosa.stft(
            filtered_audio,
            n_fft=4096,
            win_length=4096,
            window='hann'
        )

        audio_tensor = torch.tensor(audio_windowed.astype(np.float32))
        stft_gt_tensor = torch.tensor(stft_gt.astype(np.complex64))
        stft_train_tensor = torch.tensor(stft_train.astype(np.complex64))

        return audio_tensor, stft_train_tensor, stft_gt_tensor, raw_audio

"""Preprocessing del dataset FMA: split, forme d'onda e STFT"""

import os
import pickle
import random
from glob import glob

import librosa
import numpy as np
from scipy import signal

from . import config


def save_on_drive(data_list, split, dir_name):
    path = config.split_dir(split)  # solleva ValueError se split non valido
    os.makedirs(path, exist_ok=True)

    print(f"SAVING {split} ({len(data_list)} elementi) ...")
    for i, data in enumerate(data_list):
        obj = f"{i}_{dir_name}.mus"
        filename = os.path.join(path, obj)
        with open(filename, "wb") as fd:
            pickle.dump(data, fd)


def build_dataset_splits(num_cart=30, min_songs=10, seed=config.SEED):
    """Carica le tracce FMA, le porta a lunghezza fissa e le divide in train/test/val."""
    random.seed(seed)
    target_length = config.TRACK_LEN

    processed = 0
    for entry in os.listdir(config.FMA_DIR):
        if processed >= num_cart:
            break
        subpath = os.path.join(config.FMA_DIR, entry)
        if not os.path.isdir(subpath):
            continue
        processed += 1

        data_list = []
        print(subpath)
        for song in os.listdir(subpath):
            try:
                data, _ = librosa.load(os.path.join(subpath, song), sr=None)
                data = librosa.util.fix_length(data, size=target_length)
                data_list.append(data)
            except Exception as e:
                print(f"Errore in directory {entry}, file: {song}. Error: {e}")

        tot_songs = len(data_list)
        if tot_songs < min_songs:
            continue

        indexes = list(range(tot_songs))
        random.shuffle(indexes)

        train_len = round(tot_songs * 0.7)
        test_len = round(tot_songs * 0.15)

        data_array = np.asarray(data_list)
        train = data_array[indexes[:train_len]]
        test = data_array[indexes[train_len : train_len + test_len]]
        validation = data_array[indexes[train_len + test_len :]]

        save_on_drive(train, "train", str(entry))
        save_on_drive(test, "test", str(entry))
        save_on_drive(validation, "validation", str(entry))


def extract_waveforms(src_dir=None):
    """Converte i .mus dello split train in forme d'onda .npy (in WAVE_DIR)."""
    src_dir = src_dir or config.TRAIN_DIR
    os.makedirs(config.WAVE_DIR, exist_ok=True)

    for filename in sorted(os.listdir(src_dir)):
        filepath = os.path.join(src_dir, filename)
        if not (os.path.isfile(filepath) and filename.endswith(".mus")):
            continue
        try:
            with open(filepath, "rb") as f:
                loaded = pickle.load(f)

            y = _extract_waveform(loaded)
            if y is None:
                print(f"Salto {filename}: tipo inatteso {type(loaded)}")
                continue

            npy_name = os.path.splitext(filename)[0] + ".npy"
            np.save(os.path.join(config.WAVE_DIR, npy_name), y)
        except Exception as e:
            print(f"Errore {filename}: {e}")


def extract_stfts(src_dir=None):
    """Calcola e salva la STFT (in STFT_DIR) di ogni .mus dello split train."""
    src_dir = src_dir or config.TRAIN_DIR
    os.makedirs(config.STFT_DIR, exist_ok=True)
    window = signal.get_window(config.WINDOW, config.WIN_LENGTH)

    for filename in sorted(os.listdir(src_dir)):
        filepath = os.path.join(src_dir, filename)
        if not (os.path.isfile(filepath) and filename.endswith(".mus")):
            continue
        try:
            with open(filepath, "rb") as f:
                loaded = pickle.load(f)

            y = _extract_waveform(loaded)
            if y is None:
                print(f"Salto {filename}: tipo inatteso {type(loaded)}")
                continue

            y = y.astype(np.float32)
            stft = librosa.stft(
                y,
                n_fft=config.N_FFT,
                win_length=config.WIN_LENGTH,
                window=window,
            )
            npy_name = os.path.splitext(filename)[0] + ".npy"
            np.save(os.path.join(config.STFT_DIR, npy_name), stft)
        except Exception as e:
            print(f"Errore nel calcolo della STFT {filename}: {e}")


def paired_wave_stft_paths(wave_dir=None, stft_dir=None):
    """Restituisce due liste (wave, stft) allineate per nome file."""
    wave_dir = wave_dir or config.WAVE_DIR
    stft_dir = stft_dir or config.STFT_DIR

    wave_names = {os.path.basename(p) for p in glob(os.path.join(wave_dir, "*.npy"))}
    stft_names = {os.path.basename(p) for p in glob(os.path.join(stft_dir, "*.npy"))}
    common = sorted(wave_names & stft_names)

    audio_paths = [os.path.join(wave_dir, n) for n in common]
    stft_paths = [os.path.join(stft_dir, n) for n in common]
    return audio_paths, stft_paths


def _extract_waveform(loaded):
    """Estrae il vettore audio da un oggetto .mus ."""
    if isinstance(loaded, tuple) and len(loaded) >= 1:
        return loaded[0]
    if isinstance(loaded, np.ndarray):
        return loaded
    return None

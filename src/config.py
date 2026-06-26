"""Configurazione centralizzata: path e iperparametri.

Tutti i path e i parametri condivisi dalla pipeline vivono qui, così basta
cambiare un solo punto per spostare il progetto da Kaggle a un'altra macchina.
I default riproducono l'ambiente Kaggle originale.
"""

import os

import torch

# ===================== PATH =====================
# Radice dove vengono scritti i dati intermedi (default ambiente Kaggle).
DATA_ROOT = os.environ.get("DATA_ROOT", "/kaggle/temp")
# Cartella di lavoro per download e checkpoint (default ambiente Kaggle).
WORKING_DIR = os.environ.get("WORKING_DIR", "/kaggle/working")

# Dataset FMA scaricato/estratto qui.
FMA_DIR = os.path.join(WORKING_DIR, "fma_small")

# Split dei brani serializzati (.mus).
TRAIN_DIR = os.path.join(DATA_ROOT, "train_v2")
TEST_DIR = os.path.join(DATA_ROOT, "test_v2")
VALIDATION_DIR = os.path.join(DATA_ROOT, "validation_v2")

# Forme d'onda (.npy) e relative STFT (.npy).
WAVE_DIR = os.path.join(DATA_ROOT, "wave")
STFT_DIR = os.path.join(DATA_ROOT, "stft")

# Dove salvare i checkpoint della GAN.
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", WORKING_DIR)


def split_dir(split):
    """Restituisce la cartella per uno split ('train'|'test'|'validation')."""
    dirs = {"train": TRAIN_DIR, "test": TEST_DIR, "validation": VALIDATION_DIR}
    if split not in dirs:
        raise ValueError(f"Split sconosciuto: {split!r}. Attesi {list(dirs)}.")
    return dirs[split]


# ===================== AUDIO / STFT =====================
SAMPLE_RATE = 22050        # sample rate degli audio
CUTOFF = 7000             # frequenza di cutoff del filtro passa-basso
TRACK_LEN = 661500        # lunghezza fissa della traccia in campioni
N_FFT = 4096
WIN_LENGTH = 4096
HOP_LENGTH = 1024
WINDOW = "hann"           # finestra STFT: UNICA per preprocessing e dataset (coerenza)
FREQ_BINS = N_FFT // 2 + 1  # 2049

# ===================== TRAINING =====================
SEED = 13

# ===================== DEVICE =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

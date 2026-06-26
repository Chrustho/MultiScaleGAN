"""Smoke test su CPU: forward G+D e uno step GAN completo con tensori random.

Non addestra nulla di reale: serve solo a verificare che la logica corretta
(scaler, ramo discriminatore, gradient penalty, feature matching sul generatore,
``del`` protetta) giri senza eccezioni e produca loss finite — senza bisogno di
Kaggle, GPU o del dataset FMA.

Uso:  python tests/smoke_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import ReteRecWithAttention
from src.discriminator import MultiScaleDiscriminator
from src.losses import CombinedLoss, DiscriminatorLoss
from src import train_gan


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    # STFT piccola per restare leggeri: [B, F, T]. F ridotto rispetto a 2049 reale.
    B, F, T = 2, 257, 64
    stft_in = torch.randn(B, F, T) + 1j * torch.randn(B, F, T)
    stft_gt = torch.randn(B, F, T) + 1j * torch.randn(B, F, T)

    G = ReteRecWithAttention(window_size=4096, freq_bins=F, hidden_size=16).to(device)
    D = MultiScaleDiscriminator(freq_bins=F, num_scales=3, use_spectral_norm=True).to(device)

    g_loss_fn = CombinedLoss(alpha=0.5, beta=0.9)
    d_loss_fn = DiscriminatorLoss(lambda_gp=0.4)
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)

    # 1) Forward del generatore
    in_real, in_imag = stft_in.real.permute(0, 2, 1).contiguous(), stft_in.imag.permute(0, 2, 1).contiguous()
    fake_real, fake_imag = G(in_real, in_imag)
    assert fake_real.shape == (B, T, F), f"shape generatore inattesa: {fake_real.shape}"
    print(f"[ok] Generatore forward -> {tuple(fake_real.shape)}")

    # 2) Forward del discriminatore
    outputs, feats = D(stft_in)
    assert len(outputs) == 3 and len(feats) == 3, "atteso output a 3 scale"
    print(f"[ok] Discriminatore forward -> {len(outputs)} scale")

    # 3) Step del discriminatore (include la gradient penalty / double-backward)
    d_dict = train_gan.discriminator_step(D, G, opt_d, d_loss_fn, stft_in, stft_gt, device)
    assert torch.isfinite(torch.tensor(d_dict["total"])), "loss D non finita"
    assert "gp" in d_dict, "gradient penalty non calcolata"
    print(f"[ok] discriminator_step -> {d_dict}")

    # 4) Step del generatore (feature matching su G, autocast disabilitato su CPU)
    g_stats = train_gan.generator_step(
        G, D, opt_g, g_loss_fn, scaler_g=None,
        stft_in=stft_in, stft_gt=stft_gt, device=device,
        use_discriminator=True, do_optim_step=True,
    )
    for k, v in g_stats.items():
        assert torch.isfinite(torch.tensor(float(v))), f"statistica G non finita: {k}={v}"
    print(f"[ok] generator_step -> {g_stats}")

    # 5) Ramo senza discriminatore (verifica che la 'del' protetta non sollevi NameError)
    g_stats_nod = train_gan.generator_step(
        G, None, opt_g, g_loss_fn, scaler_g=None,
        stft_in=stft_in, stft_gt=stft_gt, device=device,
        use_discriminator=False, do_optim_step=True,
    )
    print(f"[ok] generator_step (senza D) -> total={g_stats_nod['total']:.4f}")

    print("\nSMOKE TEST SUPERATO: nessuna eccezione, tutte le loss finite.")


if __name__ == "__main__":
    main()

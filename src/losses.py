"""Funzioni di loss per la GAN.

Versione canonica (CORRETTA) che sostituisce le definizioni duplicate sparse nel
notebook (celle 27, 28 e dentro la cella 34). Correzioni principali:
  * ``CombinedLoss`` restituisce (totale, mse, coerenza) e NON stampa nulla;
  * la **feature matching loss** è un obiettivo del GENERATORE: qui è esposta come
    funzione usata dal training di G, e NON è più sommata alla loss del
    discriminatore (com'era erroneamente nell'originale);
  * ``DiscriminatorLoss`` calcola solo loss avversaria (+ gradient penalty opz.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCoherenceLoss(nn.Module):
    """Penalizza differenze nella variazione temporale del modulo della STFT."""

    def forward(self, real_pred, imag_pred, real_target, imag_target):
        assert real_pred.shape == imag_pred.shape, f"{real_pred.shape} != {imag_pred.shape}"
        assert real_target.shape == imag_target.shape, f"{real_target.shape} != {imag_target.shape}"
        assert real_pred.shape == real_target.shape, f"pred {real_pred.shape} != target {real_target.shape}"

        stft_pred = torch.complex(real_pred, imag_pred)
        stft_target = torch.complex(real_target, imag_target)

        pred_diff = stft_pred[:, :, 1:] - stft_pred[:, :, :-1]
        target_diff = stft_target[:, :, 1:] - stft_target[:, :, :-1]

        return F.mse_loss(pred_diff.abs(), target_diff.abs())


class CombinedLoss(nn.Module):
    """Loss di ricostruzione del generatore: MSE complessa + coerenza spettrale."""

    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.spectral_coherence_loss = SpectralCoherenceLoss()

    def forward(self, real_pred, imag_pred, real_target, imag_target):
        mse_loss = F.mse_loss(real_pred, real_target) + F.mse_loss(imag_pred, imag_target)
        coherence_loss = self.spectral_coherence_loss(real_pred, imag_pred, real_target, imag_target)
        total_loss = self.alpha * mse_loss + self.beta * coherence_loss
        return total_loss, mse_loss, coherence_loss


def feature_matching_loss(feats_real, feats_fake):
    """Feature matching tra feature reali (detached) e fake del discriminatore.

    ``feats_*`` sono liste per-scala, ognuna lista di tensori per-layer
    (formato restituito da ``MultiScaleDiscriminator``). È un obiettivo del
    GENERATORE: le feature reali vanno passate già senza gradiente.
    """
    loss = 0.0
    for scale_real, scale_fake in zip(feats_real, feats_fake):
        for fr, ff in zip(scale_real, scale_fake):
            loss = loss + torch.mean(torch.abs(fr.detach() - ff))
    return loss


def generator_adv_loss(disc_fake_outputs):
    """Loss avversaria WGAN per il generatore: -mean su ogni scala."""
    loss = 0.0
    for fake_out in disc_fake_outputs:
        loss = loss + (-torch.mean(fake_out))
    return loss


class DiscriminatorLoss(nn.Module):
    """Loss del discriminatore: avversaria WGAN + gradient penalty opzionale."""

    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def adversarial_loss(self, disc_real, disc_fake, loss_type="wgan"):
        if loss_type == "lsgan":
            real_target = 0.9
            loss_real = torch.mean((disc_real - real_target) ** 2)
            loss_fake = torch.mean(disc_fake ** 2)
        elif loss_type == "wgan":
            loss_real = -torch.mean(disc_real)
            loss_fake = torch.mean(disc_fake)
        else:
            loss_real = F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real))
            loss_fake = F.binary_cross_entropy_with_logits(disc_fake, torch.zeros_like(disc_fake))
        return loss_real + loss_fake

    def gradient_penalty(self, discriminator, real_stft, fake_stft):
        batch_size = real_stft.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=real_stft.device)
        interpolated = alpha * real_stft + (1 - alpha) * fake_stft
        interpolated.requires_grad_(True)
        disc_interpolated, _ = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=disc_interpolated[0].sum(),
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    def forward(self, discriminator, real_stft, fake_stft, use_gp=False):
        disc_real_outputs, _ = discriminator(real_stft)
        disc_fake_outputs, _ = discriminator(fake_stft.detach())

        adv_loss = 0.0
        for dr, df in zip(disc_real_outputs, disc_fake_outputs):
            adv_loss = adv_loss + self.adversarial_loss(dr, df, "wgan")

        if use_gp:
            gp = self.gradient_penalty(discriminator, real_stft, fake_stft)
            total_loss = adv_loss + self.lambda_gp * gp
            losses_dict = {"adv_loss": adv_loss.item(), "gp": gp.item()}
        else:
            total_loss = adv_loss
            losses_dict = {"adv_loss": adv_loss.item()}

        return total_loss, losses_dict

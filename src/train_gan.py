"""Addestramento della GAN."""

import gc
import os

import torch

from . import config
from .discriminator import MultiScaleDiscriminator
from .generator import ReteRecWithAttention
from .losses import (
    CombinedLoss,
    DiscriminatorLoss,
    feature_matching_loss,
    generator_adv_loss,
)

BATCH_SIZE = 2
ACCUMULATION_STEPS = 3
EPOCHS = 30

LR_G = 2e-3
LR_D = 2e-3 * 0.3

ALPHA = 0.5  # peso MSE nella loss di ricostruzione
BETA = 0.9  # peso coerenza spettrale
ADV_WEIGHT = 0.7  # peso loss avversaria del generatore
LAMBDA_FEAT = 1.0  # peso feature matching (sul generatore)
LAMBDA_GP = 0.4  # peso gradient penalty (sul discriminatore)

D_STEPS = 2
D_WARMUP_EPOCHS = 3
D_STEPS_WARMUP = 3

USE_DISCRIMINATOR = True
CHECKPOINT_EVERY = 5


def _move_obj_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device)
        except Exception:
            return obj
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _move_obj_to_device(v, device)
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_obj_to_device(x, device) for x in obj)
    return obj


def move_optimizer_state_to_device(opt, device):
    for _param_id, state in list(opt.state.items()):
        if isinstance(state, dict):
            for k, v in list(state.items()):
                state[k] = _move_obj_to_device(v, device)


def clear_gpu_memory(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        print(
            f"Stato memoria GPU: Totale {total:.2f}GB, Allocata {allocated:.2f}GB, Libera {total - allocated:.2f}GB"
        )
    else:
        gc.collect()
        print("Esecuzione su CPU: memoria CUDA non disponibile.")


def _to_generator_input(stft_complex):
    """Da STFT complessa [B, F, T] alle parti (real, imag) in [B, T, F] per il generatore."""
    return stft_complex.real.permute(0, 2, 1).contiguous(), stft_complex.imag.permute(
        0, 2, 1
    ).contiguous()


def _to_discriminator_stft(real_btf, imag_btf):
    """Da parti (real, imag) in [B, T, F] alla STFT complessa [B, F, T] per il discriminatore."""
    return torch.complex(
        real_btf.permute(0, 2, 1).contiguous(),
        imag_btf.permute(0, 2, 1).contiguous(),
    )


def discriminator_step(D, G, opt_d, d_loss_fn, stft_in, stft_gt, device):
    """Un singolo aggiornamento del discriminatore (fp32). Ritorna il dizionario delle loss."""
    in_real, in_imag = _to_generator_input(stft_in)

    with torch.no_grad():
        fake_real, fake_imag = G(in_real, in_imag)
    fake_stft = _to_discriminator_stft(fake_real, fake_imag)
    real_stft = stft_gt  # già complessa [B, F, T]

    opt_d.zero_grad()
    d_loss, d_dict = d_loss_fn(D, real_stft, fake_stft, use_gp=True)
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
    opt_d.step()

    d_dict["total"] = d_loss.item()
    return d_dict


def generator_step(
    G,
    D,
    opt_g,
    g_loss_fn,
    scaler_g,
    stft_in,
    stft_gt,
    device,
    use_discriminator,
    do_optim_step,
):
    """Forward+backward del generatore con accumulazione. Ritorna le statistiche di loss."""
    in_real, in_imag = _to_generator_input(stft_in)
    gt_real, gt_imag = _to_generator_input(stft_gt)  # [B, T, F]

    if use_discriminator and D is not None:
        for p in D.parameters():
            p.requires_grad = False
        D.eval()

    g_adv_val = 0.0
    g_feat_val = 0.0
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        fake_real, fake_imag = G(in_real, in_imag)

        if fake_real.shape != gt_real.shape:
            min_t = min(fake_real.shape[1], gt_real.shape[1])
            min_f = min(fake_real.shape[2], gt_real.shape[2])
            fake_real, fake_imag = (
                fake_real[:, :min_t, :min_f],
                fake_imag[:, :min_t, :min_f],
            )
            gt_real, gt_imag = gt_real[:, :min_t, :min_f], gt_imag[:, :min_t, :min_f]

        g_rec_total, g_mse, g_coherence = g_loss_fn(
            fake_real, fake_imag, gt_real, gt_imag
        )

        if use_discriminator and D is not None:
            fake_stft_g = _to_discriminator_stft(fake_real, fake_imag)
            real_stft = stft_gt

            fake_outputs_g, fake_feats_g = D(fake_stft_g)
            with torch.no_grad():
                _, real_feats = D(real_stft)

            g_adv = generator_adv_loss(fake_outputs_g)
            g_feat = feature_matching_loss(real_feats, fake_feats_g)
            g_adv_val, g_feat_val = g_adv.detach().item(), g_feat.detach().item()

            g_loss = (
                g_rec_total + ADV_WEIGHT * g_adv + LAMBDA_FEAT * g_feat
            ) / ACCUMULATION_STEPS
        else:
            g_loss = g_rec_total / ACCUMULATION_STEPS

    if scaler_g is not None:
        scaler_g.scale(g_loss).backward()
    else:
        g_loss.backward()

    if use_discriminator and D is not None:
        for p in D.parameters():
            p.requires_grad = True
        D.train()

    if do_optim_step:
        if scaler_g is not None:
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_g.step()
        opt_g.zero_grad()

    return {
        "total": g_loss.item() * ACCUMULATION_STEPS,
        "mse": g_mse.item(),
        "coherence": g_coherence.item(),
        "adv": g_adv_val,
        "feat": g_feat_val,
    }


def _new_history():
    return {
        "g_total_loss": [],
        "g_mse_loss": [],
        "g_coherence_loss": [],
        "g_adv_loss": [],
        "d_total_loss": [],
        "d_adv_loss": [],
    }


def main():
    import numpy as np
    from torch.utils.data import DataLoader, random_split
    from tqdm import tqdm

    from .dataset import AudioSTFTDataset
    from .preprocessing import paired_wave_stft_paths

    device = config.DEVICE
    clear_gpu_memory(device)
    print(f"Dispositivo: {device}")

    if device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    # Dataset
    audio_files, stft_files = paired_wave_stft_paths()
    if not audio_files:
        raise ValueError(
            f"Nessun file accoppiato trovato in {config.WAVE_DIR} / {config.STFT_DIR}"
        )
    print(f"Trovate {len(audio_files)} coppie audio/STFT")

    dataset = AudioSTFTDataset(audio_files, stft_files)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=gen
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin
    )
    print(f"Addestramento: {len(train_dataset)}, Validazione: {len(val_dataset)}")

    # Verifica formato dati, deve essere [B, freq_bins, time] (BFT)
    _, stft_in, stft_gt, _ = next(iter(train_loader))
    if stft_in.shape[1] != config.FREQ_BINS:
        raise ValueError(
            f"Formato dati inatteso: atteso [B, {config.FREQ_BINS}, T], ottenuto {tuple(stft_in.shape)}"
        )
    freq_bins = config.FREQ_BINS

    # Modelli
    G = ReteRecWithAttention(
        window_size=config.WIN_LENGTH, freq_bins=freq_bins, hidden_size=128
    )
    print(f"Parametri generatore: {sum(p.numel() for p in G.parameters()):,}")

    D, opt_d, d_loss_fn = None, None, None
    if USE_DISCRIMINATOR:
        D = MultiScaleDiscriminator(
            freq_bins=freq_bins, num_scales=3, use_spectral_norm=True
        )
        print(f"Parametri discriminatore: {sum(p.numel() for p in D.parameters()):,}")
        opt_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
        d_loss_fn = DiscriminatorLoss(lambda_gp=LAMBDA_GP)

    opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    g_loss_fn = CombinedLoss(alpha=ALPHA, beta=BETA)

    scaler_g = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = _new_history()
    start_epoch = 0

    # Checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "gan_checkpoint_resume.pt")
    if os.path.exists(checkpoint_path):
        print(f"Caricamento checkpoint da {checkpoint_path} ...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"Attenzione: map_location={device} fallito, ricaduta su CPU: {e}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")

        if "G_state" in ckpt:
            G.load_state_dict(ckpt["G_state"], strict=False)
            print("Generatore: pesi caricati.")
        if USE_DISCRIMINATOR and D is not None and "D_state" in ckpt:
            D.load_state_dict(ckpt["D_state"], strict=False)
            print("Discriminatore: pesi caricati.")
        if "opt_g" in ckpt:
            try:
                opt_g.load_state_dict(ckpt["opt_g"])
                move_optimizer_state_to_device(opt_g, device)
            except Exception as e:
                print("Warning: impossibile caricare opt_g:", e)
        if USE_DISCRIMINATOR and opt_d is not None and "opt_d" in ckpt:
            try:
                opt_d.load_state_dict(ckpt["opt_d"])
                move_optimizer_state_to_device(opt_d, device)
            except Exception as e:
                print("Warning: impossibile caricare opt_d:", e)
        if scaler_g is not None and "scaler_g" in ckpt:
            try:
                scaler_g.load_state_dict(ckpt["scaler_g"])
            except Exception as e:
                print("Warning: impossibile caricare scaler_g:", e)
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Ripresa dall'epoca {start_epoch}.")
    else:
        print(f"Nessun checkpoint in {checkpoint_path} — training da zero.")

    G.to(device)
    if D is not None:
        D.to(device)

    print("\n=== Avvio addestramento ===")
    print(
        f"Batch size: {BATCH_SIZE}, Accumulation steps: {ACCUMULATION_STEPS} (effettivo {BATCH_SIZE * ACCUMULATION_STEPS})"
    )
    print(f"Uso discriminatore: {USE_DISCRIMINATOR}")

    for epoch in range(start_epoch, EPOCHS):
        G.train()
        if D is not None:
            D.train()

        epoch_g = {"total": 0.0, "mse": 0.0, "coherence": 0.0, "adv": 0.0}
        epoch_d = {"total": 0.0, "adv_loss": 0.0}

        d_steps = D_STEPS if epoch >= D_WARMUP_EPOCHS else D_STEPS_WARMUP
        pbar = tqdm(train_loader, desc=f"Epoca {epoch + 1}/{EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            if batch_idx % 10 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

            _, stft_in, stft_gt, _ = batch
            stft_in = stft_in.to(device)
            stft_gt = stft_gt.to(device)

            # Discriminatore
            if USE_DISCRIMINATOR and D is not None:
                for _ in range(d_steps):
                    d_dict = discriminator_step(
                        D, G, opt_d, d_loss_fn, stft_in, stft_gt, device
                    )
                epoch_d["total"] += d_dict["total"]
                epoch_d["adv_loss"] += d_dict["adv_loss"]

            # Generatore
            do_step = (batch_idx + 1) % ACCUMULATION_STEPS == 0
            g_stats = generator_step(
                G,
                D,
                opt_g,
                g_loss_fn,
                scaler_g,
                stft_in,
                stft_gt,
                device,
                USE_DISCRIMINATOR,
                do_step,
            )
            for k in epoch_g:
                epoch_g[k] += g_stats[k]

            postfix = {
                "Gen": f"{epoch_g['total'] / (batch_idx + 1):.4f}",
                "MSE": f"{epoch_g['mse'] / (batch_idx + 1):.4f}",
                "Coh": f"{epoch_g['coherence'] / (batch_idx + 1):.4f}",
            }
            if USE_DISCRIMINATOR:
                postfix["Disc"] = f"{epoch_d['total'] / (batch_idx + 1):.4f}"
            if device.type == "cuda":
                postfix["Mem"] = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
            pbar.set_postfix(postfix)

        n = max(1, len(train_loader))
        print(f"\n=== Riepilogo Epoca {epoch + 1} ===")
        print(
            f"Generatore - Totale: {epoch_g['total'] / n:.4f}, MSE: {epoch_g['mse'] / n:.4f}, Coerenza: {epoch_g['coherence'] / n:.4f}"
        )
        history["g_total_loss"].append(epoch_g["total"] / n)
        history["g_mse_loss"].append(epoch_g["mse"] / n)
        history["g_coherence_loss"].append(epoch_g["coherence"] / n)
        if USE_DISCRIMINATOR:
            print(f"Generatore - Avversario: {epoch_g['adv'] / n:.4f}")
            print(
                f"Discriminatore - Totale: {epoch_d['total'] / n:.4f}, Adv: {epoch_d['adv_loss'] / n:.4f}"
            )
            history["g_adv_loss"].append(epoch_g["adv"] / n)
            history["d_total_loss"].append(epoch_d["total"] / n)
            history["d_adv_loss"].append(epoch_d["adv_loss"] / n)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            _run_validation(G, g_loss_fn, val_loader, device)
            _save_checkpoint(epoch + 1, G, D, opt_g, opt_d, scaler_g, history)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    _save_checkpoint(EPOCHS, G, D, opt_g, opt_d, scaler_g, history, final=True)
    np.savez(os.path.join(config.CHECKPOINT_DIR, "training_history.npz"), **history)
    print("\n=== Addestramento completato ===")


def _run_validation(G, g_loss_fn, val_loader, device, max_batches=5):
    G.eval()
    val_total = val_mse = val_coh = 0.0
    n = min(max_batches, len(val_loader))
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n:
                break
            _, stft_in, stft_gt, _ = batch
            stft_in = stft_in.to(device)
            stft_gt = stft_gt.to(device)
            in_real, in_imag = _to_generator_input(stft_in)
            gt_real, gt_imag = _to_generator_input(stft_gt)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                fake_real, fake_imag = G(in_real, in_imag)
                v_total, v_mse, v_coh = g_loss_fn(
                    fake_real, fake_imag, gt_real, gt_imag
                )
            val_total += v_total.item()
            val_mse += v_mse.item()
            val_coh += v_coh.item()
    G.train()
    print(
        f"Validazione ({n} batch): Totale {val_total / max(1, n):.4f}, MSE {val_mse / max(1, n):.4f}, Coerenza {val_coh / max(1, n):.4f}"
    )


def _save_checkpoint(epoch, G, D, opt_g, opt_d, scaler_g, history, final=False):
    name = f"gan_checkpoint_{'final' if final else f'epoch_{epoch}'}.pt"
    path = os.path.join(config.CHECKPOINT_DIR, name)
    ckpt = {
        "epoch": epoch,
        "G_state": G.state_dict(),
        "opt_g": opt_g.state_dict(),
        "history": history,
    }
    if D is not None:
        ckpt["D_state"] = D.state_dict()
        ckpt["opt_d"] = opt_d.state_dict()
    if scaler_g is not None:
        ckpt["scaler_g"] = scaler_g.state_dict()
    torch.save(ckpt, path)
    print(f"Checkpoint salvato: {path}")


if __name__ == "__main__":
    main()

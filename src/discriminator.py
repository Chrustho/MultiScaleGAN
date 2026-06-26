"""Discriminatore multi-scala per la GAN di super-resolution audio."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubDiscriminator(nn.Module):
    """Sotto-discriminatore per una singola scala (processa reale + immaginario)."""

    def __init__(self, channels_mult=1, use_spectral_norm=True):
        super().__init__()

        base_channels = 32
        channels = [
            base_channels * channels_mult,
            base_channels * 2 * channels_mult,
            base_channels * 4 * channels_mult,
            base_channels * 8 * channels_mult,
            base_channels * 8 * channels_mult,
        ]

        def norm_layer(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        # Encoder per parte reale
        self.real_conv1 = norm_layer(
            nn.Conv2d(1, channels[0], kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))
        )
        self.real_conv2 = norm_layer(
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=(5, 3),
                stride=(2, 2),
                padding=(2, 1),
            )
        )
        self.real_conv3 = norm_layer(
            nn.Conv2d(
                channels[1],
                channels[2],
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            )
        )
        self.real_conv4 = norm_layer(
            nn.Conv2d(
                channels[2],
                channels[3],
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            )
        )

        # Encoder per parte immaginaria
        self.imag_conv1 = norm_layer(
            nn.Conv2d(1, channels[0], kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))
        )
        self.imag_conv2 = norm_layer(
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=(5, 3),
                stride=(2, 2),
                padding=(2, 1),
            )
        )
        self.imag_conv3 = norm_layer(
            nn.Conv2d(
                channels[1],
                channels[2],
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            )
        )
        self.imag_conv4 = norm_layer(
            nn.Conv2d(
                channels[2],
                channels[3],
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            )
        )

        # Fusione delle feature reali e immaginarie
        self.fusion_conv = norm_layer(
            nn.Conv2d(
                channels[3] * 2,
                channels[4],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
        )

        # Layer finale per il punteggio di discriminazione
        self.final_conv = norm_layer(
            nn.Conv2d(channels[4], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

        # Normalizzazione per-campione (compatibile con la gradient penalty)
        self.norm1 = nn.InstanceNorm2d(channels[0], affine=True)
        self.norm2 = nn.InstanceNorm2d(channels[1], affine=True)
        self.norm3 = nn.InstanceNorm2d(channels[2], affine=True)
        self.norm4 = nn.InstanceNorm2d(channels[3], affine=True)
        self.norm_fusion = nn.InstanceNorm2d(channels[4], affine=True)

        self.dropout = nn.Dropout2d(0.2)

    def forward(self, stft_complex):
        real = stft_complex.real  # [batch, freq_bins, time_frames]
        imag = stft_complex.imag

        real = real.transpose(1, 2)  # [batch, time, freq]
        imag = imag.transpose(1, 2)

        if real.dim() == 3:
            real = real.unsqueeze(1)  # [batch, 1, time, freq]
        if imag.dim() == 3:
            imag = imag.unsqueeze(1)

        # Parte reale
        r1 = F.leaky_relu(self.norm1(self.real_conv1(real)), 0.2)
        r2 = F.leaky_relu(self.norm2(self.real_conv2(r1)), 0.2)
        r3 = F.leaky_relu(self.norm3(self.real_conv3(r2)), 0.2)
        r4 = F.leaky_relu(self.norm4(self.real_conv4(r3)), 0.2)
        r4 = self.dropout(r4)

        # Parte immaginaria
        i1 = F.leaky_relu(self.norm1(self.imag_conv1(imag)), 0.2)
        i2 = F.leaky_relu(self.norm2(self.imag_conv2(i1)), 0.2)
        i3 = F.leaky_relu(self.norm3(self.imag_conv3(i2)), 0.2)
        i4 = F.leaky_relu(self.norm4(self.imag_conv4(i3)), 0.2)
        i4 = self.dropout(i4)

        combined = torch.cat([r4, i4], dim=1)

        fused = F.leaky_relu(self.norm_fusion(self.fusion_conv(combined)), 0.2)
        output = self.final_conv(fused)

        # Feature intermedie per la feature matching loss
        return output, [r1, r2, r3, r4, i1, i2, i3, i4, fused]


class MultiScaleDiscriminator(nn.Module):
    """Discriminatore multi-scala: 3 sotto-discriminatori a risoluzioni diverse."""

    def __init__(self, freq_bins, num_scales=3, use_spectral_norm=True):
        super().__init__()
        self.num_scales = num_scales
        self.freq_bins = freq_bins

        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            channels_mult = 2**i
            self.discriminators.append(
                SubDiscriminator(channels_mult, use_spectral_norm)
            )

        # Downsampling temporale tra le scale
        self.downsample = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

    def forward(self, stft_complex):
        outputs = []
        features = []

        current_input = stft_complex

        for i, disc in enumerate(self.discriminators):
            output, feat = disc(current_input)
            outputs.append(output)
            features.append(feat)

            # Downsample per la scala successiva (tranne l'ultima)
            if i < self.num_scales - 1:
                real = current_input.real.transpose(1, 2).unsqueeze(
                    1
                )  # [batch, 1, time, freq]
                imag = current_input.imag.transpose(1, 2).unsqueeze(1)

                real = (
                    self.downsample(real).squeeze(1).transpose(1, 2)
                )  # [batch, freq, time]
                imag = self.downsample(imag).squeeze(1).transpose(1, 2)

                current_input = torch.complex(real, imag)

        return outputs, features

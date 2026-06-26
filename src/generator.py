"""Rete generatrice.

CODICE DELL'ALTRO TESISTA — copiato VERBATIM dalla cella 24 del notebook
originale. Non modificato: eventuali problemi sono documentati in REVIEW.md.
(Solo l'header di import è stato reso esplicito per usarlo come modulo; la
variabile globale ``device`` usata internamente da forward è preservata.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------- MODELLO ---------------
class ReteRecWithAttention(nn.Module):
    def __init__(self, window_size, freq_bins, hidden_size=128, dropout_p=0.2):
        super(ReteRecWithAttention, self).__init__()
        self.window_size = window_size
        self.freq_bins = freq_bins

        self.rec_imag_1 = nn.LSTM(self.freq_bins, hidden_size*2, num_layers=1, batch_first=True)
        self.rec_real_1 = nn.LSTM(self.freq_bins, hidden_size*2, num_layers=1, batch_first=True)

        self.rec_imag_2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=1, batch_first=True)
        self.rec_real_2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=1, batch_first=True)

        self.rec_dec_1 = nn.LSTM(hidden_size*2, hidden_size*2, num_layers=1, batch_first=True)
        self.rec_dec_2 = nn.LSTM(hidden_size*2, hidden_size*2, num_layers=1, batch_first=True)

        self.proiezione_imag_1 = nn.Linear(hidden_size*2, self.freq_bins//2)
        self.proiezione_imag_2 = nn.Linear(self.freq_bins//2, self.freq_bins)
        self.proiezione_real_1 = nn.Linear(hidden_size*2, self.freq_bins//2)
        self.proiezione_real_2 = nn.Linear(self.freq_bins//2, self.freq_bins)

        self.init_layers()

    def init_layers(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'rec_' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'rec_' in name and 'weight_ih' in name:
                    nn.init.orthogonal_(param.data)
                elif 'proiezione' in name:
                    nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, real, imag):
        real = real.to(device)
        imag = imag.to(device)

        out_imag = F.leaky_relu(self.rec_imag_2(F.leaky_relu(self.rec_imag_1(imag)[0]))[0])
        out_real = F.leaky_relu(self.rec_real_2(F.leaky_relu(self.rec_real_1(real)[0]))[0])

        encoder_output = torch.cat([out_imag, out_real], dim=2)

        outputs_real = torch.zeros(real.size(1), real.size(0), real.size(2)).to(device)
        outputs_imag = torch.zeros(imag.size(1), imag.size(0), imag.size(2)).to(device)

        decoder_input = encoder_output[:, -1, :].squeeze(1) if encoder_output.dim() > 2 else encoder_output[:, -1, :]

        for t in range(real.size(1)):
            if decoder_input.dim() == 1:
                decoder_input = decoder_input.unsqueeze(0)
            if decoder_input.dim() == 2:
                decoder_input = decoder_input.unsqueeze(1)

            decoder_output_1 = F.leaky_relu(self.rec_dec_1(decoder_input)[0])
            decoder_output = F.leaky_relu(self.rec_dec_2(decoder_output_1)[0])

            if decoder_output.dim() == 3:
                decoder_output = decoder_output.squeeze(1)
            decoder_output = decoder_output + encoder_output[:, t, :]

            out_proj_real = F.leaky_relu(self.proiezione_real_2(F.leaky_relu(self.proiezione_real_1(decoder_output))))
            out_proj_imag = F.leaky_relu(self.proiezione_imag_2(F.leaky_relu(self.proiezione_imag_1(decoder_output))))

            outputs_real[t] = out_proj_real
            outputs_imag[t] = out_proj_imag
            decoder_input = decoder_output.unsqueeze(1)

        outputs_real = outputs_real.permute(1, 0, 2)
        outputs_imag = outputs_imag.permute(1, 0, 2)

        return outputs_real, outputs_imag

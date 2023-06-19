import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.k = args.top_k

        self.conv = nn.Sequential(Inception_Block_V1(args.d_model, args.d_ff, num_kernels=args.num_kernels),
                                  nn.GELU(),
                                  Inception_Block_V1(args.d_ff, args.d_model, num_kernels=args.num_kernels))

    def forward(self, x):
        B, T, N = x.size()
        period_list, p_weight = FFT_for_Period(x, self.k)
        res = []

        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.pred_len + self.seq_len) // period + 1) * period
                padding = torch.zeros([B, length-self.pred_len-self.seq_len, N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).coutinous()
            # 2D conv
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len+self.pred_len), :])
        res = torch.stack(res, dim=-1)  # bs * length * c * k
        # adaptive aggregation
        p_weight = F.softmax(p_weight, dim=-1)  # bs * k
        p_weight = p_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)  # bs * length * c * k
        res = torch.sum(p_weight * res, dim=-1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.model = nn.ModuleList([TimesBlock(args) for _ in range(args.e_layers)])
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        self.layer = args.e_layers
        self.layer_norm = nn.LayerNorm(args.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                args.d_model, args.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                args.d_model, args.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(args.dropout)
            self.projection = nn.Linear(
                args.d_model * args.seq_len, args.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # z-score for x_enc
        means = x_enc.mean(1, keep_dim=True).detach()
        x_enc = x_enc - means
        std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= std

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # bs * seq_len * c
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)  # bs * seq_len+pred_len * c

        # de-norm
        # std[:, 0, :] bs * c
        # std[:, 0, :].unsqueeze(1) bs * 1 * c
        # dec_out bs * seq_len+pred_len * c
        dec_out = std * std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len+self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len+self.seq_len, 1)

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        pass

    def classification(self, x_enc, x_mark_enc):
        pass

    def anomaly_detection(self, x_enc):
        pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None






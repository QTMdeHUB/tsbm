import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TiDEResidualBlock(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.2, use_layer_norm=True, use_dropout=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        self.linear_res = nn.Linear(d_in, d_out)
        self.use_layer_norm = use_layer_norm
        self.use_dropout = use_dropout

    def forward(self, x):
        # 只最输入的最后一维进行操作：d_in ----> d_out
        # hidden
        hidden = F.relu(self.linear1(x))
        hidden = self.dropout(self.linear2(hidden)) if self.use_dropout else self.linear2(hidden)
        # res
        res = self.linear_res(x)
        # layer norm
        if self.use_layer_norm:
            out = self.layer_norm(res + hidden)
        else:
            out = res + hidden

        return out


class Encoder(nn.Module):
    def __init__(self, enc_layers, d_hidden, seq_len, pred_len, r, covar_hidden, dropout, r_hat=4):
        super().__init__()
        self.pred_len = pred_len
        self.residualblock_covar = TiDEResidualBlock(r, covar_hidden, r_hat, dropout)
        self.residualblock_1 = TiDEResidualBlock((seq_len+(seq_len+pred_len)*r_hat+1), d_hidden, d_hidden, dropout)
        self.residualblock_other = nn.ModuleList(
                TiDEResidualBlock(d_hidden, d_hidden, d_hidden, dropout) for _ in range(enc_layers-1))


    def forward(self, x, covar, a):
        # x: (bs*c) * seq_len
        # covar: (bs*c) * (seq_len+pred_len) * r
        # a: (bs*c) * 1
        covar = self.residualblock_covar(covar)  # (bs*c) * (seq_len+pred_len) * r_hat
        covar_future = covar[:, -self.pred_len:, :]  # (bs*c) * pred_len * r_hat
        covar = covar.flatten(start_dim=1)  # (bs*c) * (seq_len+pred_len) * r_hat ----> bs * (seq_len+pred_len)
        enc_in = torch.cat([x, a, covar], dim=1)  # (bs*c) * (seq_len + 1 + (seq_len+pred_len)*r_hat)
        print(enc_in)

        enc_out = self.residualblock_1(enc_in)  # (bs*c) * d_hidden
        for layer in self.residualblock_other:
            enc_out = layer(enc_out)  # (bs*c) * d_hidden

        return enc_out, covar_future


class Decoder(nn.Module):
    def __init__(self, dec_layers, d_hidden, pred_len, p, d_tem_hidden, dropout, r_hat=4):
        super().__init__()
        # dense decoder
        self.pred_len = pred_len
        self.residualblock = nn.ModuleList(TiDEResidualBlock(d_hidden, d_hidden, d_hidden, dropout)
                                           for _ in (dec_layers-1))
        self.residualblock.append(TiDEResidualBlock(d_hidden, d_hidden, pred_len*p, dropout))

        # temporal decoder
        self.residualblock_temporal = TiDEResidualBlock(p+r_hat, d_tem_hidden, 1, dropout)

    def forward(self, x, covar_future):
        # x: (bs*c) * d_hidden
        # covar_future: (bs*c) * pred_len * r_hat
        # dense decoder
        for layer in self.residualblock:
            dec_out = layer(x)  # (bs*c) * (pred_len*p)
        dec_out = rearrange(dec_out, 'b (h p) -> b h p')  # (bs*c) * pred_len * p
        # temporal decoder
        temporal_dec_in = torch.cat([dec_out, covar_future], dim=-1)  # (bs*c) * pred_len * (p+r_hat)
        temporal_dec_out = self.residualblock_temporal(temporal_dec_in)

        return temporal_dec_out  # (bs*c) * pred_len * 1


class Model(nn.Module):
    """
    TiDE Model
    """
    def __init__(self, args, r):
        # r: 协变量的维度
        super().__init__()
        self.args = args
        self.task_name = args.task_name

        if self.task_name == 'long_term_forecast' or 'shot_term_forecast':
            self.encoder = Encoder(args.e_layers, args.d_model, args.seq_len, args.pred_len, r,
                                   args.covar_hidden, args.dropout)
            self.decoder = Decoder(args.d_layers, args.d_model, args.pred_len, args.p, args.temporalDecoderHidden, args.dropout)
            self.global_skip = nn.Linear(args.seq_len, args.pred_len)

    def forward(self, x, covar, a):
        # x: bs * seq_len * c
        # covar: bs * (seq_len+pred_len) * c * r
        # a: bs * c * 1
        x = rearrange(x, 'b l c -> (b c) l')
        covar = rearrange(covar, 'b l c r -> (b c) l r')
        a = rearrange(a, 'b c 1 -> (b c) 1')

        # encoder
        enc_out, covar_future = self.encoder(x, covar, a)  # (bs*c) * d_hidden, (bs*c) * pred_len * r_hat

        # decoder
        dec_out = self.decoder(enc_out, covar_future)  # (bs*c) * pred_len * 1

        # global skip connection
        out = dec_out.squeeze(-1) + self.global_skip(x)
        out = rearrange(out, '(b n) l -> b l n')  # bs * pred_len * c

        return out


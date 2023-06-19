import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer Model
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, args):
        super().__init__()
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # decomp
        self.decomp = series_decomp(args.moving_avg)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq,
                                                  args.dropout)

        # Encoder Layer
        enc_layer = EncoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                            output_attention=args.output_attention
                            ),
                            args.d_model, args.n_heads
                        ),
                        args.d_model,
                        args.d_ff,
                        kernel_size=args.moving_avg,
                        dropout=args.dropout,
                        activation=args.activation
                    )

        # Encoder
        self.encoder = Encoder(
            [enc_layer for l in range(args.e_layers)],
            norm_layer=my_Layernorm(args.d_model)
        )

        # Decoder Layer
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(args.dec_in, args.d_model, args.embed, args.freq,
                                                      args.dropout)
            dec_layer = DecoderLayer(
                            AutoCorrelationLayer(
                                AutoCorrelation(True, args.factor, attention_dropout=args.dropout,
                                            output_attention=False
                                ),
                                args.d_model, args.n_heads
                            ),
                            AutoCorrelationLayer(
                                AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                            output_attention=False
                                ),
                                args.d_model, args.n_heads
                            ),
                            args.d_model,
                            args.c_out,
                            d_ff=args.d_ff,
                            kernel_size=args.moving_avg,
                            dropout=args.dropout,
                            activation=args.activation
                        )
            self.decoder = Decoder(
                                [dec_layer for l in range(args.d_layers)],
                                norm_layer=my_Layernorm(args.d_model),
                                projection=nn.Linear(args.d_model, args.c_out, bias=True)
                            )

        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                args.d_model, args.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                args.d_model, args.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(args.dropout)
            self.projection = nn.Linear(
                args.d_model * args.seq_len, args.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: bs * seq_len * c
        """
        # process decoder input
        mean_place_holder = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)  # 对应论文中的O
        zero_place_holder = torch.zeros([x_dec[0], self.pred_len, x_dec[2]], device=x_enc.device)  # O
        seasonal_init, trend_init = series_decomp(x_enc[:, -self.label_len:, :])  # I/2
        seasonal_init = torch.cat([seasonal_init, zero_place_holder], dim=1)  # I/2 + O = seq_len/2 + pred_len
        trend_init = torch.cat([trend_init, mean_place_holder], dim=1)  # I/2 + O

        # encoder
        # enc_out: bs * seq_len * d_model
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # 这里enc_out要变成bs * (seq_len/2 + pred_len) * d_model ????????
        enc_out = self.encoder(enc_out, attn_mask=None)

        # decoder
        # 注意这里只有seasonal_init进了embedding，因此trend_init.shape: bs * (seq_len/2 + pred_len) * c
        # dec_out.shape: bs * (seq_len/2 + pred_len) * d_model
        dec_out = self.enc_embedding(seasonal_init, x_mark_dec)
        # season_part, trend_part: bs * (seq_len/2 + pred_len) * c
        season_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # final
        dec_out = season_part + trend_part

        return dec_out  # bs * (seq_len/2 + pred_len) * c

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        pass

    def anomaly_detection(self, x_enc):
        pass

    def classification(self, x_enc, x_mark_enc):
        pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None





if __name__ == '__main__':
    x = torch.arange(10)
    x_fft = torch.fft.rfft(x)
    print(x_fft)



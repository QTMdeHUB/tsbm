import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class FlattenHead(nn.Module):
    def __init__(self, nf, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: bs * c * d_model * num_patch
        x = self.dropout(self.linear(self.flatten(x)))
        return x


class Model(nn.Module):
    """
    PatchTST Model
    """
    def __init__(self, args):
        super().__init__()

        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.patch_embeddings = PatchEmbedding(args.d_model, args.patch_len, args.stride, args.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention),
                        args.n_heads, args.d_model),
                    args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation
                ) for _ in range(args.e_layers)
            ],
            norm_layer = nn.LayerNorm(args.d_model)
        )

        self.head_nf = args.d_model * (int((args.seq_len - args.patch_len) / args.stride + 2))

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.flatten_head = FlattenHead(self.head_nf, args.pred_len, args.dropout)
        else:
            pass

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # patch
        x_enc = x_enc.permute(0, 2, 1)  # bs * c * seq_len
        enc_out, num_vars = self.patch_embeddings(x_enc)  # bs * c * num_patch * d_model
        # encode
        enc_out, attn = self.encoder(enc_out)  # bs * c * num_patch * d_model
        enc_out = torch.reshape(
            enc_out, (-1, num_vars, enc_out.shape[-2], enc_out.shape[-1]))  # bs * c * num_patch * d_model
        enc_out = enc_out.permute(0, 1, 3, 2)  # bs * c * num_patch * d_model

        # flatten head
        dec_out = self.flatten_head(enc_out)  # bs * c * pred_len
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return  dec_out[:, -self.pred_len:, :]  # bs * pred_len * c
        else:
            pass

        return None 


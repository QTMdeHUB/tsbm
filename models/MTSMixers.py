import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalMLP(nn.Module):
    def __init__(self, temporal_in, hidden=512):
        super().__init__()
        self.linear1 = nn.Linear(temporal_in, hidden)
        self.linear2 = nn.Linear(hidden, temporal_in)

    def forward(self, x):
        # x.shape: bs * s * c * slice_length
        out = nn.GELU(self.linear1(x))
        out = self.linear2(out)
        return out

class ChannelMLP(nn.Module):
    def __init__(self, c_in, m):
        super().__init__()
        self.linear1 = nn.Linear(c_in, m)
        self.linear2 = nn.Linear(m, c_in)

    def forward(self, x):
        # x.shape: bs * s * slice_len * c_in
        out = nn.GELU(self.linear1(x))
        out = self.linear2(out)
        return out


class MTS_Mixers_Block(nn.Module):
    def __init__(self, s, slice_len, c_in, m):
        super().__init__()
        self.s = s
        self.slice_len = slice_len
        self.temporal_mlp = TemporalMLP(self.slice_len)
        self.channel_mlp = ChannelMLP(c_in, m)

    def forward(self, x):
        # x : bs * seq_len(after padding) * c
        mts_in = torch.empty([x.shape[0], 0, self.slice_len, x.shape[2]], device=x.device)
        for i in range(self.s):
            x_i = x_enc[:, i::self.s, :].unsqueeze(1)
            mts_in = torch.cat([mts_in, x_i], dim=1)

        # mts_in.shape: bs * s * slice_len * c
        # temporal mlp
        temporal_out = self.temporal_mlp(mts_in.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        # merge for the shape of channel_in is same as x  ???????? how to merge???
        channel_in = temporal_out

        # channel mlp
        channel_out = self.channel_mlp(channel_in + x) + temporal_out

        return channel_out


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.s = args.s
        self.mix_mode = args.mix_mode
        self.task_name = args.task_name
        self.e_layers = args.e_layers
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.d_model = args.d_model  # channel hidden dim ------ m
        self.c_in = args.c_in # channel input dim
        self.slice_len = args.seq_len // self.s + 1  # temporal input dim

        if self.mix_mode == 'factorized':
            self.MTS_Mixers = nn.ModuleList([MTS_Mixers_Block(self.s, self.slice_len, self.c_in, self.d_model)
                                            for _ in range(self.e_layers)])
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: bs * seq_len * c
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev  # x_enc: bs * seq_len * c

        # padding & down sample  padding 取0？
        if self.seq_len % self.s != 0:
            length = self.slice_len * self.s
            padding = torch.zeros([x_enc.shape[0], length - self.seq_len, x_enc.shape[2]], device=x_enc.device)
            x_enc = torch.cat([x_enc, padding], dim=1)

        # mts layers
        # mts in: bs * seq_len(after padding) * c
        # mts out: bs * seq_len * c
        mts_out = self.MTS_Mixers(x_enc)

        # prediction
        out = self.projection(mts_out.permute(0, 2, 1)).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        # stdev: bs * 1 * c
        out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return out # bs * pred_len * c

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            pass

        return None


if __name__ == '__main__':
    # x = torch.arange(60).reshape(2, 3, 10).float()
    # x_prime = F.interpolate(x, size=5, mode='linear', align_corners=False)
    # print(x_prime, x_prime.shape)

    # # down sample
    # input_sequence = torch.arange(20).reshape(2, 10).unsqueeze(-1).repeat(1, 1, 3).float()
    # print(f'input_sequence: \n{input_sequence}')
    #
    # # 下采样因子
    # downsample_factor = 2
    #
    # # 用torch下采样
    # down_1 = input_sequence[:, ::downsample_factor, :].unsqueeze(1)
    # down_2 = input_sequence[:, 1::downsample_factor, :].unsqueeze(1)
    # # down_list = torch.cat([down_1, down_2])
    # down_list = torch.cat([down_1, down_2], dim=1)
    # temporal_in = down_list.permute(0, 1, 3, 2)
    #
    # print('+--------------------------------------+')
    # print(f'down_1: \n{down_1}')
    # print(f'down_1.shape: \n{down_1.shape}')
    # print('+--------------------------------------+')
    # print(f'down_2: \n{down_2}')
    # print(f'down_2.shape: \n{down_2.shape}')
    # print('+--------------------------------------+')
    # print(f'down_list: \n{down_list}')
    # print(f'down_list.shape: \n{down_list.shape}')
    # print('+--------------------------------------+')
    # print(f'temporal_in: \n{temporal_in}')
    # print(f'temporal_in.shape: \n{temporal_in.shape}')


    # a = torch.tensor([1, 2, 3])
    # b = a.copy()
    # a = torch.cat([a, torch.zeros(3)])
    # print(a)
    # print(b)
    x_enc = torch.arange(24).reshape(2, 3, 4).float()
    temporal_in = torch.empty([x_enc.shape[0], 0, x_enc.shape[2]])
    print(temporal_in)
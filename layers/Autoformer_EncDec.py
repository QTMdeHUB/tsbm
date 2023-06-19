import torch
import torch.nn as nn
import torch.nn.functional as F

class my_Layernorm(nn.Module):
    """
    special layer_norm designed for seasonal part
    """

    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)  # channels: d_model

    def forward(self, x):
        x_hat = self.layernorm(x)
        # torch.mean(x_hat, dim=1): bs * d_model
        # unsqueeze(1): bs * 1 * d_model
        # repeat: bs * seq_len * d_model
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 在x的两端都padding
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # padding的内容就是每个序列（样本）的第一个timestamp的值。
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        x_t = self.moving_avg(x)
        x_s = x - x_t
        return x_s, x_t


class EncoderLayer(nn.Module):
    """

    """
    def __init__(self, attention, d_model, d_ff=None, kernel_size=25, dropout=0.1, activation="relu"):
        super().__init__()
        self.attention = attention  # auto-correlation
        self.dropout = nn.Dropout(dropout)
        self.decomp1 = series_decomp(kernel_size)
        self.decomp2 = series_decomp(kernel_size)
        self.d_ff = d_ff or 4 * d_model  # dim of fcn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.activation = F.relu() if activation == 'relu' else F.gelu()

    def forward(self, x, attn_mask=None):
        # auto-corelation
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        # residual connection
        x = x + self.dropout(new_x)
        # series decpomp
        x, _ = self.decomp1(x)  # x.shape: bs * l * d_model

        # feed forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # ??为什么要transpose
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # residual + series decomp
        res, _ = self.decomp2(y + x)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer Encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # attn_layers = encoder layers
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self,x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 kernel_size=25, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.dropout = nn.Dropout(dropout)
        self.decomp1 = series_decomp(kernel_size)
        self.decomp2 = series_decomp(kernel_size)
        self.decomp3 = series_decomp(kernel_size)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)  # kernel_size=3, stride=1, padding=1保持了输出长度不变

    def forward(self, x, cross, x_mask, cross_mask):
        # residual + auto-corelation
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        # decomp1
        x, trend1 = self.decomp1(x)

        # residual + cross auto-corelation
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        # decomp2
        x, trend2 = self.decomp2(x)

        # feed forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # decomp3
        x, trend3 = self.decom3(x + y)

        # residual trend for the next decoder layer
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        # x: bs * (seq_len/2 + pred_len) * d_model
        # residual_trend: bs * (seq_len/2 + pred_len) * c(c_out)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer Decoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        # trend: bs * (seq_len/2 + pred_len) * c
        # x: bs * (seq_len/2 + pred_len) * d_model
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            # layer norm
            x = self.norm(x)

        if self.projection is not None:
            # 把x的最后一维：d_model -----> c_out
            x = self.projection(x)

        return x, trend  # 两个返回值的shape都为bs * (seq_len/2 + pred_len) * c


if __name__ == '__main__':
    # bs = 2
    # len_seq = 3
    # num_variate = 4
    # ln = my_Layernorm(num_variate)
    # ln_2 = nn.LayerNorm(num_variate)
    #
    # x = torch.rand([bs,len_seq,num_variate])
    #
    # z = ln_2(x)
    # y = ln(x)
    # bias = torch.mean(z, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
    #
    # print(y)
    # print(z)
    # print(bias)
    # print('-------------------------------------------------')
    #
    # a = torch.arange(24).reshape(2, 3, 4).float()
    # b = torch.mean(a, dim=1)
    # c = b.unsqueeze(1)
    # d = c.repeat(1, a.shape[1], 1)
    #
    # for i in [a, b, c, d]:
    #     print(i)
    #
    # print(a.shape[1])
    #
    # print( 7 // 2)

    # a = torch.arange(12).reshape(2, 2, 3)
    # print(a)
    # print(a[:, 0:1, :])
    # print(a[:, 0:1, :].repeat(1, 5, 1))
    # print(a.transpose(-1, 1))
    a = torch.rand([2, 3, 4, 6])
    b = torch.rand([2, 3, 4, 6])

    af = torch.fft.rfft(a, dim=-1)
    bf = torch.fft.rfft(b, dim=-1)

    ff = af * torch.conj(bf)
    corr = torch.fft.ifft(ff, dim=-1)

    print(af)
    print(af.shape)
    print(torch.conj(bf).shape)
    # print(corr)

    mean_ff = torch.mean(corr, dim=1)
    mean_ff_ff = torch.mean(mean_ff, dim=1)
    print(mean_ff.shape)
    print(mean_ff_ff.shape)
import torch
import torch.nn as nn

class Inception_Block_V1(nn.Module):
    def __init__(self, input_channel, output_channel, num_kernels=6, init_weight=True):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.num_kernels = num_kernels
        kernels = []
        for i in range(num_kernels):
            kernels.append(nn.Conv2d(input_channel, output_channel, kernel_size=2*i+1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)

        return res


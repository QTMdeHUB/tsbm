import torch
import numpy as np

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]  # 用于mask注意力分数。注意力分数的shape: B * H * L * L(S)
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


if __name__ == '__main__':
    fix_seed = 1222
    torch.manual_seed(fix_seed)
    B = 2
    L = 3
    H = 4
    E = 5
    attn_mask = TriangularCausalMask(B, L)

    q = torch.rand([B, L, H, E])
    k = torch.rand([B, L, H, E])

    scores = torch.einsum('blhe, bshe -> bhls', q, k)
    # print(scores)
    # print(scores.shape)  # B * H * L * L

    scores.masked_fill_(attn_mask.mask, -np.inf)
    print(scores)
    print(attn_mask.mask)
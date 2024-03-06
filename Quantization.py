import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

y_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32).T
y_table = nn.Parameter(torch.from_numpy(y_table))
y_table = y_table.to(device)

c_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32).T
c_table = nn.Parameter(torch.from_numpy(c_table))
c_table = c_table.to(device)


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    sign = torch.ones_like(x)
    sign[torch.floor(x) % 2 == 0] = -1
    y = sign * torch.cos(x * torch.pi) / 2
    out = torch.round(x) + y - y.detach()
    return out

def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.

class Quantization(nn.Module):
    def __init__(self, quality=90):
        super(Quantization, self).__init__()
        self.y_table = c_table 
        self.rounding = diff_round
    def forward(self, x):
        tmp = (self.y_table * self.factor) / 255.0
        tmp = tmp.contiguous().view(1, -1, 1, 1)
        ans = torch.zeros_like(x)
        ans[:, 0, :, :] = x[:, 0, :, :]
        ans[:, 1:, :, :] = self.rounding(x[:, 1:, :, :].float() / tmp[:, 1:, :, :])
        #ans = self.rounding(x.float() / tmp)
        return ans
    def inverse(self, x):
        tmp = (self.y_table * self.factor) / 255.0
        tmp = tmp.contiguous().view(1, -1, 1, 1)
        ans = torch.zeros_like(x)
        ans[:, 0, :, :] = x[:, 0, :, :]
        ans[:, 1:, :, :] = x[:, 1:, :, :].float() * tmp[:, 1:, :, :]
        #ans = x.float() * tmp
        return ans
    def set_quality(self, quality):
        self.factor = quality_to_factor(quality)

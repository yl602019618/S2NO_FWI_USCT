"""Original Gaussian model smoothing with replicated boundaries."""

import torch
import math
import torch.nn.functional as F

def create_gaussian_kernel(kernel_size=3, sigma=1.0):
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = x / x.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d
def blur(G, sigma = 2.0):
    G = torch.tensor(G).cuda()
    kernel_size = 2*math.ceil(2*sigma)+1
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel.expand(1, 1, -1, -1)  # 使其适用于2D卷积
    # 将张量转换为具有批处理维度和通道维度的格式
    G = G.unsqueeze(0).unsqueeze(0).float()

    padding = kernel_size // 2
    G_padded = F.pad(G, (padding, padding, padding, padding), mode='replicate')
    # 使用F.conv2d进行高斯模糊
    G_blurred = F.conv2d(G_padded, gaussian_kernel.cuda()).squeeze(0).squeeze(0)
    return G_blurred.detach().cpu()

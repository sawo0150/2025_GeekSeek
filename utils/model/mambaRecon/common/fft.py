# mambaRecon/common/fft.py
import torch

def ifft2c(x, dim=(-2, -1), img_shape=None):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim=dim)

def fft2c(x, dim=(-2, -1), img_shape=None):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim=dim)

import numpy as np
import torch


def signal_wave(batch_size, amp, f, fs, phi, t):
    Ts = 1 / fs
    n = t / Ts
    n = torch.arange(n).unsqueeze(0).repeat((batch_size, 1))
    y = amp * torch.sin(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y


def inter_wave(batch_size, amp, f, fs, phi, t):
    Ts = 1 / fs
    n = t / Ts
    n = torch.arange(n).unsqueeze(0).repeat((batch_size, 1))
    y = amp * torch.sin(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y


def raw_wave(s_wave, i_wave, i_scales, n_scale=1):
    return s_wave + i_wave * i_scales + n_scale * torch.randn(s_wave.shape)


if __name__ == "__main__":
    sig = signal_wave(1000, 1, 20, 10, 0, 1)
    inter = inter_wave(1000, 1, 700, 10, torch.pi / 4, 1)
    raw = raw_wave(sig, inter, torch.rand((2, 1)))
    print(sig, inter, raw)

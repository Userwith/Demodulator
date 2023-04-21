import os.path

import numpy as np
import torch
from modules.utils import get_hparams, progress_bar
import pandas as pd


class Simulator:

    def __init__(self, hps):
        self.batch_size = hps.simulator.batch_size
        self.t = hps.simulator.t
        self.fs = hps.simulator.fs
        self.signal_amp = hps.simulator.signal_amp
        self.signal_f = hps.simulator.signal_f
        self.signal_phi = hps.simulator.signal_phi
        self.inter_amp = hps.simulator.inter_amp
        self.inter_f = hps.simulator.inter_f
        self.inter_phi = hps.simulator.inter_phi
        self.num_i_scale = hps.simulator.num_i_scale
        self.num_n_scale = hps.simulator.num_n_scale
        self.num_train_data = hps.simulator.num_train_data
        self.num_eval_data = hps.simulator.num_eval_data
        self.num_test_data = hps.simulator.num_test_data
        self.datasets_path = hps.simulator.datasets_path
        self.n_scale_max = hps.simulator.n_scale_max
        self.i_scale_max = hps.simulator.i_scale_max
        self.i_scale_list = torch.arange(0, self.i_scale_max, self.i_scale_max / self.num_i_scale)
        self.n_scale_list = torch.arange(0, self.n_scale_max, self.n_scale_max / self.num_n_scale)
        self.step = torch.arange(self.t * self.fs)

    def signal_wave(self):
        y = self.signal_amp * torch.sin(
            2 * torch.pi * self.signal_f * self.step * (1 / self.fs) + self.signal_phi * (torch.pi / 180))
        return y.unsqueeze(0).repeat((self.batch_size, 1))

    def inter_wave(self):
        y = self.inter_amp * torch.sin(
            2 * torch.pi * self.inter_f * self.step * (1 / self.fs) + self.inter_phi * (torch.pi / 180))
        return y.unsqueeze(0).repeat((self.batch_size, 1))

    def raw_wave(self, s_wave, i_wave, i_scales, n_scale):
        return s_wave + torch.mul(i_wave, i_scales) + n_scale * torch.randn(s_wave.shape)

    def _write(self, batch_id, i_scale, n_scale, data_type, s_wave, i_wave, r_wave):
        data_path = self.datasets_path + "i_" + str(int(100*i_scale.cpu().detach().numpy())) + "_n_" + str(
            int(100*n_scale.cpu().detach().numpy())) + "/" + data_type + "/"
        is_exists = os.path.exists(data_path)
        if not is_exists:
            os.makedirs(data_path)
        s_wave = s_wave.unsqueeze(2)
        i_wave = i_wave.unsqueeze(2)
        r_wave = r_wave.unsqueeze(2)
        data = torch.concatenate((s_wave, i_wave), dim=2)
        data = torch.concatenate((data, r_wave), dim=2)
        torch.save(data, data_path + str(batch_id) + ".pth")

    def __call__(self, *args, **kwargs):
        for i, idx in zip(self.i_scale_list, range(self.i_scale_list.shape[0])):
            for n, idx2 in zip(self.n_scale_list, range(self.n_scale_list.shape[0])):
                for batch in range(int(np.floor(self.num_train_data / self.batch_size))):
                    s_wave = self.signal_wave()
                    i_wave = self.inter_wave()
                    i_scale = i * torch.rand((self.batch_size, 1))
                    n_scale = n
                    r_wave = self.raw_wave(s_wave, i_wave, i_scale, n_scale)
                    self._write(batch, i, n, "train", s_wave, i_wave, r_wave)
                for batch in range(int(np.floor(self.num_eval_data / self.batch_size))):
                    s_wave = self.signal_wave()
                    i_wave = self.inter_wave()
                    i_scale = i * torch.rand((self.batch_size, 1))
                    n_scale = n
                    r_wave = self.raw_wave(s_wave, i_wave, i_scale, n_scale)
                    self._write(batch, i, n, "eval", s_wave, i_wave, r_wave)
                for batch in range(int(np.floor(self.num_test_data / self.batch_size))):
                    s_wave = self.signal_wave()
                    i_wave = self.inter_wave()
                    i_scale = i * torch.rand((self.batch_size, 1))
                    n_scale = n
                    r_wave = self.raw_wave(s_wave, i_wave, i_scale, n_scale)
                    self._write(batch, i, n, "test", s_wave, i_wave, r_wave)
                progress_bar(idx*self.n_scale_list.shape[0]+idx2+1, self.i_scale_list.shape[0]*self.n_scale_list.shape[0])


if __name__ == "__main__":
    hps = get_hparams()
    sim = Simulator(hps)
    sim()

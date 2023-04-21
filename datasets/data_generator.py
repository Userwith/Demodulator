import os.path

import numpy as np
import torch


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
        self.train_datasets_path = hps.simulator.train_datasets_path
        self.eval_datasets_path = hps.simulator.eval_datasets_path
        self.test_datasets_path = hps.simulator.test_datasets_path
        self.n_scale_max = hps.simulator.n_scale_max
        self.i_scale_max = hps.simulator.i_scale_max
        self.i_scale_list = torch.arange(0, self.i_scale_max, self.num_i_scale)
        self.n_scale_list = torch.arange(0, self.n_scale_max, self.num_n_scale)
        self.data_type = {"train": self.train_datasets_path, "eval": self.eval_datasets_path,
                          "test": self.test_datasets_path}
        self.n = torch.arange(self.t * self.fs).unsqueeze(0).repeat((self.batch_size, 1))
    def signal_wave(self):
        y = self.signal_amp * torch.sin(2 * torch.pi * self.signal_f * self.n * (1/self.fs) + self.signal_phi * (torch.pi / 180))
        return y

    def inter_wave(self):
        y = self.inter_amp * torch.sin(2 * torch.pi * self.inter_f * self.n * (1/self.fs) + self.inter_phi * (torch.pi / 180))
        return y

    def raw_wave(self, s_wave, i_wave, i_scales, n_scale):
        return s_wave + i_wave * i_scales + n_scale * torch.randn(s_wave.shape)

    def _write(self, batch_id, i_scale, n_scale, data_type, s_wave, i_wave, r_wave):
        data_path = self.data_type[data_type] + "i_" + str(i_scale) + "_n_" + str(n_scale)
        is_exists = os.path.exists(data_path)
        if not is_exists:
            os.makedirs(data_path.decode("utf-8"))
        torch.save(s_wave, "s_" + str(batch_id) + '.pth')
        torch.save(i_wave, "i_" + str(batch_id) + '.pth')
        torch.save(r_wave, "r_" + str(batch_id) + '.pth')

    def __call__(self, *args, **kwargs):
        for i in self.i_scale_list:
            for n in self.n_scale_list:
                for batch in range(int(np.floor(self.num_train_data / self.batch_size))):
                    s_wave = self.signal_wave()
                    i_wave = self.inter_wave()
                    i_scale = i * torch.rand(self.batch_size)
                    n_scale = n
                    r_wave = self.raw_wave(s_wave, i_wave, i_scale, n_scale)
                    self._write(batch, i, n, "train", s_wave, i_wave, r_wave)
                for batch in range(int(np.floor(self.num_eval_data / self.batch_size))):
                    s_wave = self.signal_wave()
                    i_wave = self.inter_wave()
                    i_scale = i * torch.rand(self.batch_size)
                    n_scale = n
                    r_wave = self.raw_wave(s_wave, i_wave, i_scale, n_scale)
                    self._write(batch, i, n, "eval", s_wave, i_wave, r_wave)
                for batch in range(int(np.floor(self.num_test_data / self.batch_size))):
                    s_wave = self.signal_wave()
                    i_wave = self.inter_wave()
                    i_scale = i * torch.rand(self.batch_size)
                    n_scale = n
                    r_wave = self.raw_wave(s_wave, i_wave, i_scale, n_scale)
                    self._write(batch, i, n, "test", s_wave, i_wave, r_wave)

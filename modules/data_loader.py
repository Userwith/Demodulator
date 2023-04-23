import time
import os
import random
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader

import utils


class WaveDataset(torch.utils.data.Dataset):
    """

    """

    def __init__(self,hps, data_type, i_scale, n_scale, num_batch=0):
        self.hps = hps
        self.data_type = data_type
        self.i_scale = str(int(i_scale*100))
        self.n_scale = str(int(n_scale*100))
        self.dataset_path = self.hps.datasets_path+"i_"+self.i_scale+"_n_"+self.n_scale+"/"+self.data_type+"/"
        self.dataset_files = utils.load_filepaths(
            self.dataset_path) if num_batch == 0 else utils.load_filepaths(
            self.dataset_path)[:num_batch]
        self.s_waves, self.i_waves, self.r_waves = self._read()


    def _read(self):
        s_waves, i_waves, r_waves = None, None, None
        for file, idx in zip(self.dataset_files, range(len(self.dataset_files))):
            raw_data = torch.load(self.dataset_path+ file)
            s_waves = raw_data[:, :, 0] if idx == 0 else torch.concatenate((s_waves, raw_data[:, :, 0]), 0)
            i_waves = raw_data[:, :, 1] if idx == 0 else torch.concatenate((i_waves, raw_data[:, :, 1]), 0)
            r_waves = raw_data[:, :, 2] if idx == 0 else torch.concatenate((r_waves, raw_data[:, :, 2]), 0)
        return s_waves.unsqueeze(1), i_waves.unsqueeze(1), r_waves.unsqueeze(1)


    def __getitem__(self, idx):
        return self.s_waves[idx],self.i_waves[idx],self.r_waves[idx]

    def __len__(self):
        return self.hps.batch_size * len(self.dataset_files)

if __name__ == "__main__":
    hps = utils.get_hparams()
    train_dataset = WaveDataset(hps.data,"train",i_scale=0,n_scale=0)
    train_loader = DataLoader(train_dataset, num_workers=10, shuffle=True,
                             batch_size=1000, pin_memory=True,
                             drop_last=False)
    for batch_idx, items in enumerate(train_loader):
        (s_waves, i_waves, r_waves) = items
        print(s_waves.shape,i_waves.shape, r_waves.shape)
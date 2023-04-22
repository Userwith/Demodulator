import json
import os
import sys

import torch


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams():
    config_save_path = os.path.join("../configs/config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def progress_bar(x, progress_max):
    s = int(x * 100 / progress_max)
    sys.stdout.write('\r')
    sys.stdout.write("Generate progress: {}%: ".format(s))
    sys.stdout.write("|")
    sys.stdout.write("▋" * s)
    sys.stdout.write(" " * int(100 - s))
    sys.stdout.write("|")
    sys.stdout.flush()


def load_filepaths(dataset_root):
    listdir = os.listdir(dataset_root)
    return listdir


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

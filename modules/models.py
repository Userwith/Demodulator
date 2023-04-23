import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from modules import utils
from modules.data_loader import WaveDataset


class CasualDilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding=1):
        super().__init__()
        self.conv1D = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False, padding='same')
        self.ignoreOutIndex = (kernel_size - 1) * dilation

    def forward(self, x):
        return self.conv1D(x)[..., :-self.ignoreOutIndex]


class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, skipConnection):
        # as b c outputsize -> skipConnection size
        out = torch.mean(skipConnection, dim=0)

        for i in range(2):
            out = self.relu(out)
            out = self.conv1d(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.casualDilatedConv1D = CasualDilatedConv1D(res_channels, res_channels, kernel_size, dilation=dilation)
        self.resConv1D = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.skipConv1D = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputX, skipSize):
        x = self.casualDilatedConv1D(inputX)
        x1 = self.tanh(x)
        x2 = self.sigmoid(x)
        x = x1 * x2
        resOutput = self.resConv1D(x)
        resOutput = resOutput + inputX[..., -resOutput.size(2):]
        skipOutput = self.skipConv1D(x)
        skipOutput = skipOutput[..., -skipSize:]
        return resOutput, skipOutput


class StackOfResBlocks(nn.Module):

    def __init__(self, stack_size, layer_size, res_channels, skip_channels, kernel_size):
        super().__init__()
        buildDilationFunc = np.vectorize(self.buildDilation)
        dilations = buildDilationFunc(stack_size, layer_size)
        self.resBlocks = []
        for s, dilationPerStack in enumerate(dilations):
            for l, dilation in enumerate(dilationPerStack):
                resBlock = ResBlock(res_channels, skip_channels, kernel_size, dilation)
                self.add_module(f'resBlock_{s}_{l}', resBlock)  # Add modules manually
                self.resBlocks.append(resBlock)

    def buildDilation(self, stack_size, layer_size):
        # stack1=[1,2,4,8,16,...512]
        dilationsForAllStacks = []
        for stack in range(stack_size):
            dilations = []
            for layer in range(layer_size):
                dilations.append(2 ** layer)
            dilationsForAllStacks.append(dilations)
        return dilationsForAllStacks

    def forward(self, x, skipSize):
        resOutput = x
        skipOutputs = []
        for resBlock in self.resBlocks:
            resOutput, skipOutput = resBlock(resOutput, skipSize)
            skipOutputs.append(skipOutput)
        return resOutput, torch.stack(skipOutputs)


class WaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stack_size, layer_size, *args, **kwargs):
        super().__init__()
        self.stack_size = stack_size
        self.layer_size = layer_size
        self.kernel_size = kernel_size
        self.casualConv1D = CasualDilatedConv1D(in_channels, in_channels, kernel_size, dilation=1)
        self.stackResBlock = StackOfResBlocks(self.stack_size, self.layer_size, in_channels, out_channels, kernel_size)
        self.denseLayer = DenseLayer(out_channels)

    def calculateReceptiveField(self):
        return np.sum([(self.kernel_size - 1) * (2 ** l) for l in range(self.layer_size)] * self.stack_size)

    def calculateOutputSize(self, x):
        return int(x.size(2)) - self.calculateReceptiveField()

    def forward(self, x):
        # x: b c t -> input data size
        x = self.casualConv1D(x)
        skipSize = self.calculateOutputSize(x)
        _, skipConnections = self.stackResBlock(x, skipSize)
        dense = self.denseLayer(skipConnections)
        return dense

class VAEDemodulator(nn.Module):
    def __init__(self,hps):
        super().__init__()
        self.hps = hps
        self.feature_extractor = WaveNet(hps.in_channels,hps.out_channels,hps.kernel_size,hps.stack_size,hps.layer_size)

    def forward(self,input_waves):
        return self.feature_extractor(input_waves)


if __name__ == "__main__":
    hps = utils.get_hparams()
    model = VAEDemodulator(hps.model)
    train_dataset = WaveDataset(hps.data,"train",i_scale=0,n_scale=0)
    train_loader = DataLoader(train_dataset, num_workers=10, shuffle=True,
                             batch_size=1000, pin_memory=True,
                             drop_last=False)
    for batch_idx, items in enumerate(train_loader):
        (s_waves, i_waves, r_waves) = items
        print(s_waves.shape,i_waves.shape, r_waves.shape)
        input_waves = torch.concatenate((r_waves,i_waves),dim=1)
        o = model(input_waves)
        print(o.shape)
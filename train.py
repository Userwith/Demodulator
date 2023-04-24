import logging

from modules import utils

import logging
import time

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
import time
import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from modules.data_loader import WaveDataset
from modules.models import VAEDemodulator

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

CONFIG_PATH = {"model_dir": "./logs/",
    "config_path": "./configs/config.json"}
def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams(CONFIG_PATH["config_path"])
    utils.add_model_dir(hps,0.6,0.2)
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '10.147.18.71'
    os.environ['MASTER_PORT'] = hps.train.port
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,0.6, 0.2))


def run(rank, n_gpus, hps, i_scale, n_scale):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # for pytorch on win, backend use gloo
    dist.init_process_group(backend= 'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    train_dataset = WaveDataset(hps.data,"train",i_scale,n_scale)
    num_workers = 5 if mp.cpu_count() > 4 else mp.cpu_count()
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=True, pin_memory=True,
                              batch_size=hps.train.batch_size)
    if rank == 0:
        eval_dataset = WaveDataset(hps.data,"eval",i_scale,n_scale)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=500, pin_memory=True,
                                 )

    net = VAEDemodulator(hps.model).cuda(rank)
    optim = torch.optim.AdamW(
        net.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "Net_i"+str(int(i_scale*100))+"_n"+str(int(n_scale*100))+"_*.pth"), net,
                                                   optim, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, net, optim, scaler,[i_scale,n_scale],
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, net, optim, scaler,[i_scale,n_scale],
                               [train_loader, None], None, None)
        scheduler.step()


def train_and_evaluate(rank, epoch, hps, net, optim, scaler, scales, loaders, logger, writers):
    train_loader, eval_loader = loaders
    mse_loss = nn.MSELoss()
    i_scale, n_scale = scales
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    net.train()
    for batch_idx, items in enumerate(train_loader):
        s_waves, i_waves, r_waves = items
        s_waves = s_waves.cuda(rank, non_blocking=True)
        i_waves = i_waves.cuda(rank, non_blocking=True)
        r_waves = r_waves.cuda(rank, non_blocking=True)
        input_waves = torch.concatenate((r_waves,i_waves),dim=1)
        with autocast(enabled=hps.train.fp16_run):
            i_scale_hat = net(input_waves)
            print()
            s_waves_hat = r_waves-i_waves*i_scale_hat
            with autocast(enabled=False):
                loss = mse_loss(s_waves,s_waves_hat)

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        grad_norm = utils.clip_grad_value_(net.parameters(), None)
        scaler.step(optim)

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim.param_groups[0]['lr']
                losses = [loss]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}")

                scalar_dict = {"loss/g/total": loss, "learning_rate": lr,
                               "grad_norm": grad_norm}
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(net, eval_loader, writer_eval)
                utils.save_checkpoint(net, optim, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "Net_i_"+str(int(i_scale*100))+"_n_"+str(int(n_scale*100))+"_"+str(global_step)+".pth"))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(net, eval_loader, writer_eval):
    net.eval()
    wave_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            s_waves, i_waves, r_waves = items
            s_waves = s_waves.cuda(0)
            i_waves = i_waves.cuda(0)
            r_waves = r_waves.cuda(0)
            s_wave_hat = net.module.infer(r_waves, i_waves)[0].squeeze(0).cpu().numpy()[:500]
            s_wave = s_waves[0].squeeze(0).cpu().numpy()[:500]
            i_wave = i_waves[0].squeeze(0).cpu().numpy()[:500]
            r_wave = r_waves[0].squeeze(0).cpu().numpy()[:500]
            ylim = (-2,2)
            wave_dict.update({
                f"net/pred_signal_wave_{batch_idx}": utils.plot_wave_to_numpy(s_wave_hat),
                f"gt/real_signal_wave_{batch_idx}": utils.plot_wave_to_numpy(s_wave),
                f"gt/real_interference_wave_{batch_idx}": utils.plot_wave_to_numpy(i_wave),
                f"gt/real_raw_wave_{batch_idx}": utils.plot_wave_to_numpy(r_wave),
                f"compare/signal_c_raw_waves_{batch_idx}": utils.plot_compare_waves_to_numpy(s_wave,r_wave,ylim),
                f"compare/real_c_pred_waves_{batch_idx}": utils.plot_compare_waves_to_numpy(s_wave,s_wave_hat,ylim)
            })

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        waves=wave_dict
    )
    net.train()


if __name__ == "__main__":
    main()

import os
from argparse import ArgumentParser
import json
import numpy as np
import yaml
import random

import torch
from torch.utils.data import DataLoader

from util.datasets import load_dataset
from lib.models import get_model_class
from util.logging import LogEntry


def run_epoch(data_loader, model, device, train=True, early_break=False):
    log = LogEntry()

    # Setting model and dataset into train/eval mode
    if train:
        model = model.train()
        data_loader.dataset.train()
    else:
        model = model.eval()
        data_loader.dataset.eval()

    for batch_idx, (states, actions, labels_dict) in enumerate(data_loader):
        states = states.to(device)
        actions = actions.to(device)
        if random.random() < 0.5:
            (B, T, D) = states.shape
            states = torch.flip(states.reshape(B, T, 2, D // 2), dims=[2]).reshape(B, T, D)
            (B, T, D) = actions.shape
            actions = torch.flip(actions.reshape(B, T, 2, D // 2), dims=[2]).reshape(B, T, D)
        # state_dim = model.config['state_dim']
        # if model.config['state_dim'] != data_loader.dataset.state_dim:
        #     states = states[:, :, :state_dim]
        #     actions = actions[:, :, :state_dim]
        labels_dict = {key: value.to(device) for key, value in labels_dict.items()}

        batch_log = model(states, actions, labels_dict, True)

        if train:
            model.optimize(batch_log.losses)

        batch_log.itemize()  # itemize here since we shouldn't need gradient information anymore
        log.absorb(batch_log)

        if early_break:
            break

    log.average(N=len(data_loader.dataset))

    print('TRAIN' if train else 'TEST')
    print(str(log))

    return log.to_dict()


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser for training')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)

    basedir = os.path.join('exp', config['log']['basedir'])
    os.makedirs(basedir, exist_ok=True)






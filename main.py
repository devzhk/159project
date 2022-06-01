import os
import sys
from argparse import ArgumentParser
import numpy as np
import random
import yaml
import torch

from utils.preprocess import normalize, transform_to_svd_components
from utils.datasets import MouseV1Dataset


def preprocess(raw_data):
    full_dataset = []
    for i, (name, sequence) in enumerate(raw_data['sequences'].items()):
        # Vectorizes each frame in the sequence
        vec_seq = sequence['keypoints'].reshape(
            sequence['keypoints'].shape[0], -1)
        full_dataset.append(normalize(vec_seq))

    full_dataset = np.concatenate(full_dataset, axis=0)
    _, svd, mean = transform_to_svd_components(full_dataset)
    preprocessed_data = []
    sub_seq_length = 21
    sliding_window = 5
    for i, (name, sequence) in enumerate(raw_data['sequences'].items()):
        # Preprocess sequences
        vec_seq = sequence['keypoints'].reshape(
            sequence['keypoints'].shape[0], -1)
        vec_seq = normalize(vec_seq)

        if i == 0:  # Here for debugging - leave to demonstrate that preprocessing works
            control = vec_seq
            control = control.reshape(control.shape[0], -1)
            control = np.pad(control, ((
                sub_seq_length//2, sub_seq_length-1-sub_seq_length//2), (0, 0)), mode='edge')
            control = np.stack([control[i:len(control)+i-sub_seq_length+1:sliding_window]
                                for i in range(sub_seq_length)], axis=1)

        vec_seq, _, _ = transform_to_svd_components(
            vec_seq,
            svd_computer=svd,
            mean=mean
        )

        # Pads the beginning and end of the sequence with duplicate frames
        vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)
        pad_vec = np.pad(vec_seq, ((sub_seq_length//2,
                                    sub_seq_length-1-sub_seq_length//2), (0, 0)), mode='edge')

        # Converts sequence into [number of sub-sequences, frames in sub-sequence, x/y alternating keypoints]
        sub_seqs = np.stack([pad_vec[i:len(pad_vec)+i-sub_seq_length+1:sliding_window]
                             for i in range(sub_seq_length)], axis=1)
        preprocessed_data.append(sub_seqs)
    preprocessed_data = np.concatenate(preprocessed_data, axis=0)
    return preprocessed_data


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser for training')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    user_train = np.load('data/user_train.npy', allow_pickle=True).item()
    preprocessed_data = preprocess(user_train)
    test_prop = 0.20
    test_len = int(len(preprocessed_data) * test_prop)
    np.random.shuffle(preprocessed_data)
    data_train = preprocessed_data[test_len:]
    data_test = preprocessed_data[:test_len]

    data_config = {
        'name': 'mouse_v1',
        'data_train': data_train,
        'data_test': data_test,
        'device': device,
    }

    dataset = MouseV1Dataset(data_config)





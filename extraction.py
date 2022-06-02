from copyreg import pickle
import os
from argparse import ArgumentParser
import json
import pickle
from tqdm import tqdm
import numpy as np
import yaml
import random

import torch
from util.datasets.mouse_v1.preprocess import transform_to_svd_components, \
    transform_svd_to_keypoints, normalize, unnormalize
from lib.models import get_model_class


def get_embedding(seq, model, device):
    '''
    Input:
        seq: seq_length x 2 x 9
    Return:
        seq_emb: seq_length x 2 x embedding_dim
        seq_recon: seq_length x 2 x 18
    '''
    batchsize = 128
    sub_seq_len = 21
    num_pad = sub_seq_len // 2
    seq = seq.reshape(seq.shape[0], -1)
    pad_seq = np.pad(seq, ((num_pad, sub_seq_len - num_pad - 1), (0, 0)), mode='edge')

    data = np.stack([pad_seq[i:pad_seq.shape[0]+ i - sub_seq_len + 1:1] for i in range(sub_seq_len)], axis=1)

    num_batch = data.shape[0] // batchsize

    tensor_data = torch.tensor(data, dtype=torch.float32)

    emb_list = []
    recon_list = []
    label_dict = {}

    model.eval()
    model.stage = 2
    with torch.no_grad():
        for i in range(num_batch + 1):
            if i == num_batch:
                states = tensor_data[batchsize * i:]
            else:
                states = tensor_data[batchsize * i: batchsize * (i + 1)]
            actions = states[:, 1:] - states[:, :-1]

            states = states.to(device)
            actions = actions.to(device)

            # embedding of first
            log, recon1, embedding1 = model(states, actions, label_dict, True)

            (B, T, D) = states.shape
            states = torch.flip(states.reshape(B, T, 2, D // 2), dims=[2]).reshape(B, T, D)
            (B, T, D) = actions.shape
            actions = torch.flip(actions.reshape(B, T, 2, D // 2), dims=[2]).reshape(B, T, D)
            log, recon2, embedding2 = model(states, actions, label_dict, True)

            # batchsize x emb_dim
            embedding = torch.stack([embedding1, embedding2], dim=1).cpu()
            # batchsize x 2 x emb_dim

            # T x batchsize x 18
            recons = torch.stack([recon1[num_pad], recon2[num_pad]], dim=0)
            # seq_len x 2 x 2 x 9
            recons = recons.permute(1, 0, 2).cpu()

            emb_list.append(embedding)
            recon_list.append(recons)

    seq_emb = torch.cat(emb_list, dim=0)
    seq_recon = torch.cat(recon_list, dim=0)
    return seq_emb.numpy(), seq_recon.numpy()


def preprocess(raw_data, svd, mean, model, device):
    for (name, sequence) in tqdm(raw_data['annotator-id_0'].items()):
        # Preprocess sequences
        raw_seq = sequence['keypoints']
        raw_seq = np.transpose(raw_seq, (0, 1, 3, 2))
        # L x 2 x 7 x 2
        vec_seq = normalize(raw_seq.reshape(raw_seq.shape[0], 1, -1))
        vec_seq = vec_seq.reshape(raw_seq.shape)
        # L x 2 x 7 x 2
        vec_seq, _, _ = transform_to_svd_components(
            vec_seq,
            svd_computer=svd,
            mean=mean,
            stack_agents=True
        )
        # L x 2 x 9
        embedding, recon = get_embedding(vec_seq, model, device)
        # recon: L x 2 x 18
        re_seq = transform_svd_to_keypoints(
            recon.reshape(-1, recon.shape[-1]),
            svd_computer=svd,
            mean=mean,
            stack_agents=True,
            stack_axis=1
        )
        re_seq = re_seq.reshape(-1, 2, 2, 7, 2)
        sequence['embedding'] = embedding
        sequence['reconstruction'] = re_seq
    return raw_data


def extract(ckpt_path, model_config, data_paths, save_name='02'):
    # load svd transformation matrix
    svd_dir = 'util/datasets/mouse_v1/svd'
    svd_path = os.path.join(svd_dir, 'calms21_svd_computer.pickle')
    mean_path = os.path.join(svd_dir, 'calms21_mean.pickle')
    with open(svd_path, 'rb') as f:
        svd_computer = pickle.load(f)
    with open(mean_path, 'rb') as f:
        svd_mean = pickle.load(f)
    # initialize model with pretrained weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(ckpt_path, map_location=device)
    print(f'Weights loaded from {ckpt_path}')

    model_class = get_model_class(model_config['name'].lower())
    model = model_class(model_config).to(device)
    model.loss_params['contrastive_loss_weight'] = 0.0
    model.loss_params['consistency_loss_weight'] = 0.0
    model.loss_params['decoding_loss_weight'] = 0.0
    model.load_state_dict(state_dict)

    # dataset
    for path in data_paths:
        raw = np.load(path, allow_pickle=True).item()
        new_data = preprocess(raw, svd_computer, svd_mean, model, device)
        save_file = path.replace('.npy', f'_{save_name}.npy')
        np.save(save_file, new_data)
        print(f'Saved at {save_file}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_dir', type=str,
                        required=False, default='',
                        help='path to all config files for experiments')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--save_name', type=str)
    args = parser.parse_args()
    # Get JSON file
    config_file = os.path.join(os.getcwd(), 'configs', args.config_dir)
    with open(config_file, 'r') as f:
        config = json.load(f)
    model_config = config['model_config']
    ckpt_path = args.ckpt
    paths = ['data/calms21_task1_train.npy', 'data/calms21_task1_test.npy']
    extract(ckpt_path, model_config, data_paths=paths, save_name=args.save_name)







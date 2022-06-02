import numpy as np
import torch

path = 'data/calms21_task1_train_02.npy'
data = np.load(path, allow_pickle=True).item()
print('debug')
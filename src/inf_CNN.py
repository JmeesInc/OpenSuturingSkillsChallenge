import os
import torch
import glob
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import augment_sequence, OSSTestDataset, PointSequenceClassifier


input_dir = sys.argv[1]
output_csv = sys.argv[2]

model_path = "/app/src/weight/CNN1D_2400_Task2.pth"
coords = sorted(glob.glob(input_dir + '/*.npy'))
print(coords)

test_dataset = OSSTestDataset(coords)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model=  PointSequenceClassifier(num_classes=4).to(device)
model.load_state_dict(torch.load(model_path))

for i, coord in enumerate(test_loader):
    coord = coord.to(device)
    with torch.cuda.amp.autocast():
        output = model(coord)
        output = {key: value.cpu().detach().numpy() for key, value in output.items()}
    if i == 0:
        final_pred = output
    else:
        for key, value in output.items():
            final_pred[key] = np.concatenate([final_pred[key], value], axis=0)

data_dict = {}
for key, value in final_pred.items():
    if key.startswith("ostats_"):
        for i in range(value.shape[1]):
            data_dict[f"{key}_{i+1}"] = value[:, i]
    else:
        for i in range(value.shape[1]):
            data_dict[f"{key}_{i+1}"] = value[:, i]


# Create DataFrame from the dictionary
pred_df = pd.DataFrame(data_dict)
pred_df.to_csv(output_csv, index=False)


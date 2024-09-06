# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class CFG:
    fold = 0
    lr = 1e-5
    epoch = 1000
    batch_size = 2
    target_len = 2400
    model_path = "/home/shunsuke/OSS/model/CNN1D_2400_5_Task2.pth"
    
    base_dir = '/data1/shared/miccai/EndoVis2024/OSS/coord/'
    label_path = '/data1/shared/miccai/EndoVis2024/OSS/OSATS.csv'

    aux_SUTURES = True
    aux_GROUP = True
    aux_TIME = True

    task = 2

# %%
def augment_sequence(sequence, target_length=2400):
    seq_len = sequence.shape[0]
    step_size = seq_len // target_length
    selected_indices = []

    for i in range(target_length):
        start = i * step_size
        end = min((i + 1) * step_size, seq_len)
        selected_index = np.random.randint(start, end)
        selected_indices.append(selected_index)

    selected_indices = np.sort(selected_indices)
    augmented_sequence = sequence[selected_indices]
    
    return augmented_sequence
# %%
import wandb
wandb.login()
wandb.init(project='OSS-1DCNN-Task1', name='CNN1D_notebook2', config={
    "fold": CFG.fold,
    "lr": CFG.lr,
    "batch_size": CFG.batch_size,
    "model_path": CFG.model_path,
    "aux_SUTURES": CFG.aux_SUTURES,
    "aux_GROUP": CFG.aux_GROUP,
    "aux_TIME": CFG.aux_TIME,
    "seq_len": CFG.target_len
})

# %%
class OSSDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=CFG.base_dir, label_df=pd.read_csv(CFG.label_path), mode="train", aux_SUTURES=False, aux_GROUP=False, aux_TIME=False):
        self.label = label_df.pivot(index="VIDEO", columns="INVESTIGATOR", values='GLOBA_RATING_SCORE')
        self.paths = [CFG.base_dir + video + ".npy" for video in self.label.index]
        #self.paths.remove("R22G.npy")
        self.aux_df = label_df.groupby("VIDEO").first()[["TIME", "SUTURES", "GROUP"]]
        self.aux_df["TIME"] = self.aux_df["TIME"].apply(lambda x: 1 if x == "PRE" else 0)
        self.aux_df["GROUP"] = self.aux_df["GROUP"].replace({"TUTOR-LED": 0, "HMD-BASED": 1, "E-LEARNING": 2})

        self.aux_SUTURES = aux_SUTURES
        self.aux_GROUP = aux_GROUP
        self.aux_TIME = aux_TIME
        self.OSTATS = label_df[["VIDEO", "OSATS_RESPECT", "OSATS_MOTION", "OSATS_INSTRUMENT", "OSATS_SUTURE", "OSATS_FLOW", "OSATS_KNOWLEDGE", "OSATS_PERFORMANCE", "OSATS_FINAL_QUALITY"]].groupby("VIDEO").mean()
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        aux = []
        coord = self.paths[idx]
        coord = np.load(coord).astype(np.float32)
        coord[:, :, 0] /= 1980
        coord[:, :, 1] /= 1080
        coord = coord.reshape(-1, 8)
        coord = augment_sequence(coord, target_length=CFG.target_len)

        score = self.label.iloc[idx].to_numpy().astype(np.float32) # 8, 10, 12
        
        # cls = 0 if label<15, 1 if 15<=label<20, 2 if 20<=labe<25, 3 if 25<=label < 30, 4 if 30<=label
        cls = np.zeros(4, dtype=np.float32)
        cls[0] = 1 if score[0] < 15.5 else 0
        cls[1] = 1 if 15.5 <= score[0] < 23.5 else 0
        cls[2] = 1 if 23.5 <= score[0] < 31.5 else 0
        cls[3] = 1 if 31.5 <= score[0] else 0

        if self.aux_SUTURES or self.aux_GROUP or self.aux_TIME:
            aux_row = self.aux_df.loc[self.label.index[idx]]

            if self.aux_SUTURES:
                suture = aux_row["SUTURES"]
                steps = np.arange(0, suture + 0.5, 0.5)
                suture = np.array([suture], dtype=np.float32)
                #各ステップを繰り返して長さが600になるようにする
                times_sutures = np.repeat(steps, coord.shape[0] // 15 // len(steps) + 1)[:CFG.target_len // 15] # 3000 / 15  = 200
                times_sutures = times_sutures.astype(np.float32)

            if self.aux_GROUP:
                group = np.eye(3)[int(aux_row["GROUP"])].astype(np.float32)
            
            if self.aux_TIME:
                time = np.eye(2)[int(aux_row["TIME"])].astype(np.float32)
            
            OSTATS = self.OSTATS.loc[self.label.index[idx]].to_numpy().astype(np.float32)
            ostats_respect = np.eye(5)[int(OSTATS[0]) - 1].astype(np.float32)
            ostats_motion = np.eye(5)[int(OSTATS[1]) - 1].astype(np.float32)
            ostats_instrument = np.eye(5)[int(OSTATS[2]) - 1].astype(np.float32)
            ostats_suture = np.eye(5)[int(OSTATS[3]) - 1].astype(np.float32)
            ostats_flow = np.eye(5)[int(OSTATS[4]) - 1].astype(np.float32)
            ostats_knowledge = np.eye(5)[int(OSTATS[5]) - 1].astype(np.float32)
            ostats_performance = np.eye(5)[int(OSTATS[6]) - 1].astype(np.float32)
            ostats_final_quality = np.eye(5)[int(OSTATS[7]) - 1].astype(np.float32)
        
        return coord,score,suture, group, time, cls, times_sutures, ostats_respect, ostats_motion, ostats_instrument, ostats_suture, ostats_flow, ostats_knowledge, ostats_performance, ostats_final_quality

# %%
df = pd.read_csv(CFG.label_path)
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for _fold, (train_idx, valid_idx) in enumerate(kf.split(df, df['GROUP'], groups=df['STUDENT'])):
    if _fold == CFG.fold:
        break

train_df = df.iloc[train_idx]
valid_df = df.iloc[valid_idx]

train_dataset = OSSDataset(label_df=train_df, mode="train", aux_SUTURES=CFG.aux_SUTURES, aux_GROUP=CFG.aux_GROUP, aux_TIME=CFG.aux_TIME)
val_dataset = OSSDataset(label_df=valid_df, mode="train", aux_SUTURES=CFG.aux_SUTURES, aux_GROUP=CFG.aux_GROUP, aux_TIME=CFG.aux_TIME)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=64, drop_last=True)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=len(val_dataset)//CFG.batch_size, drop_last=False)

del df, kf, train_idx, valid_idx
gc.collect()
    

# %%
coord,score,suture, group, time, cls, times_sutures, ostats_respect, ostats_motion, ostats_instrument, ostats_suture, ostats_flow, ostats_knowledge, ostats_performance, ostats_final_quality = next(iter(train_loader))

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super(ResidualBlock, self).__init__()
        self.res = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.SELU()
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, x):
        residual = self.res(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + residual
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

# %%
class PointSequenceClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=256, num_layers=5, num_classes=5, dropout=0.3):
        super(PointSequenceClassifier, self).__init__()

        # conv layer for generalization (large kernel size, large stride but padding)
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, 16, kernel_size=31, stride=5, padding=15), nn.BatchNorm1d(16), nn.SELU(), nn.Dropout(dropout))
        self.residual_1 = ResidualBlock(16, 16, kernel_size=21, stride=1, padding=10, dropout=dropout)
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 11, 3, 5), nn.BatchNorm1d(32), nn.SELU(), nn.Dropout(dropout))
        self.residual_2 = ResidualBlock(32, 32, 5, 1, 2, dropout=dropout)
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.SELU(), nn.Dropout(dropout))
        # Bidirectional GRU
        self.gru = nn.GRU(input_size=64,  # 4 points each with (x, y) coordinates
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        # time regression: output (B, seq_len, 1)
        self.time_regression = nn.Conv1d(512, 1, kernel_size=7, stride=1, padding=3)
        # Attention layer
        self.attention =nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=8)
        # Fully connected layers with residual connections
        self.conv4 = nn.Sequential(nn.Conv1d(CFG.target_len // 15, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv5 = ResidualBlock(128, 32, 1, 1, 0, dropout=dropout)
        self.conv6 = nn.Sequential(nn.Conv1d(32, 1, 1), nn.Flatten())
        self.lin2 = nn.Sequential(nn.Linear(512, 1024), nn.SELU())
        self.score_head = nn.Linear(1024, 3)
        self.suture_head = nn.Linear(1024, 1)
        self.group_head = nn.Linear(1024, 3)
        self.time_head = nn.Linear(1024, 2)
        self.classifier = nn.Linear(1024, 4)

        self.ostats_respect = nn.Linear(1024, 5)
        self.ostats_motion = nn.Linear(1024, 5)
        self.ostats_instrument = nn.Linear(1024, 5)
        self.ostats_suture = nn.Linear(1024, 5)
        self.ostats_flow = nn.Linear(1024, 5)
        self.ostats_knowledge = nn.Linear(1024, 5)
        self.ostats_performance = nn.Linear(1024, 5)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pass through residual blocks
        x = x.permute(0, 2, 1)  # (B, seq_len, 8) -> (B, 8, seq_len)
        x=  self.conv1(x)
        x = self.residual_1(x)
        x = self.conv2(x)
        x = self.residual_2(x)
        x = self.conv3(x)
        
        x = x.permute(0, 2, 1)  # (B, seq_len, 64)
        # Pass through GRU
        gru_out, hidden = self.gru(x)  # gru_out: (B, seq_len, hidden_dim * 2)
        time_reg = self.time_regression(gru_out.permute(0, 2, 1)).squeeze()  # time_reg: (B, seq_len, 1)

        context, attn_weights = self.attention(gru_out, gru_out, gru_out)
        # Fully connected layers with residual connections
        out = self.conv4(context)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.lin2(out)
        
        score = self.score_head(out)
        suture = self.suture_head(out)
        group = self.group_head(out)
        time = self.time_head(out)
        cls = self.classifier(out)

        ostats_respect = self.ostats_respect(out)
        ostats_motion = self.ostats_motion(out)
        ostats_instrument = self.ostats_instrument(out)
        ostats_suture = self.ostats_suture(out)
        ostats_flow = self.ostats_flow(out)
        ostats_knowledge = self.ostats_knowledge(out)
        ostats_performance = self.ostats_performance(out)

        return {"score": score, "suture": suture, "group": group, "time": time, "class": cls, "time_reg": time_reg, "ostats_respect": ostats_respect, "ostats_motion": ostats_motion, "ostats_instrument": ostats_instrument, "ostats_suture": ostats_suture, "ostats_flow": ostats_flow, "ostats_knowledge": ostats_knowledge, "ostats_performance": ostats_performance}

# Example usage
model = PointSequenceClassifier(num_classes=4)  # Adjust num_classes as needed
output = model(coord)
for key, val in output.items():
    print(f"{key}: {val.shape}")

# %%
optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
scaler = torch.cuda.amp.GradScaler()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epoch, eta_min=1e-7)

model = model.to(device)


score_loss = nn.L1Loss()
suture_loss = nn.L1Loss()
group_loss = nn.BCEWithLogitsLoss()
time_loss = nn.BCEWithLogitsLoss()
class_loss = nn.BCEWithLogitsLoss()
time_reg_loss = nn.MSELoss()

def loss_function(output, target, epoch):
    loss = 0
    scr = score_loss(output["score"], target["score"])
    sut = suture_loss(output["suture"], target["suture"])
    grp = group_loss(output["group"], target["group"])
    tim = time_loss(output["time"], target["time"])
    cls = class_loss(output['class'], target['class'])
    tim_reg = time_reg_loss(output['time_reg'], target['time_reg'])
    if epoch > CFG.epoch * 4 //5:
        loss += cls
    else:
        loss = scr + sut + grp + tim + cls + tim_reg
    wandb.log({"score_loss": scr, "suture_loss": sut, "group_loss": grp, "time_loss": tim, "class_loss": cls, "time_reg_loss": tim_reg})
    return loss

def ostats_loss(output, target):
    loss = 0
    ostats_respect = F.binary_cross_entropy_with_logits(output["ostats_respect"], target["ostats_respect"])
    ostats_motion = F.binary_cross_entropy_with_logits(output["ostats_motion"], target["ostats_motion"])
    ostats_instrument = F.binary_cross_entropy_with_logits(output["ostats_instrument"], target["ostats_instrument"])
    ostats_suture = F.binary_cross_entropy_with_logits(output["ostats_suture"], target["ostats_suture"])
    ostats_flow = F.binary_cross_entropy_with_logits(output["ostats_flow"], target["ostats_flow"])
    ostats_knowledge = F.binary_cross_entropy_with_logits(output["ostats_knowledge"], target["ostats_knowledge"])
    ostats_performance = F.binary_cross_entropy_with_logits(output["ostats_performance"], target["ostats_performance"])
    loss = ostats_respect + ostats_motion + ostats_instrument + ostats_suture + ostats_flow + ostats_knowledge + ostats_performance
    wandb.log({"ostats_respect": ostats_respect, "ostats_motion": ostats_motion, "ostats_instrument": ostats_instrument, "ostats_suture": ostats_suture, "ostats_flow": ostats_flow, "ostats_knowledge": ostats_knowledge, "ostats_performance": ostats_performance})
    return loss

from sklearn.metrics import f1_score, average_precision_score

def calc_metrics(output, target):
    metrics = {}
    metrics["score_MAE"] = np.mean(np.abs(output["score"] - target["score"]))
    metrics["suture_MAE"] = np.mean(np.abs(output["suture"] - target["suture"]))
    metrics["group_AP"] = average_precision_score(target["group"], 1 / (1 + np.exp(-output["group"])), average='macro')
    metrics["time_AP"] = average_precision_score(target["time"], 1 / (1 + np.exp(-output["time"])), average='macro')
    metrics["class_AP"] = average_precision_score(target["class"], 1 / (1 + np.exp(-output["class"])), average='macro')
    metrics["time_reg_MSE"] = np.mean((output["time_reg"] - target["time_reg"]) ** 2)
    metrics["F1"] = f1_score(np.argmax(target["class"], axis=1), np.argmax(output["class"], axis=1), average='macro')
    metrics["ostats_respect_F1"] = f1_score(np.argmax(target["ostats_respect"], axis=1), np.argmax(output["ostats_respect"], axis=1), average='macro')
    metrics["ostats_motion_F1"] = f1_score(np.argmax(target["ostats_motion"], axis=1), np.argmax(output["ostats_motion"], axis=1), average='macro')
    metrics["ostats_instrument_F1"] = f1_score(np.argmax(target["ostats_instrument"], axis=1), np.argmax(output["ostats_instrument"], axis=1), average='macro')
    metrics["ostats_suture_F1"] = f1_score(np.argmax(target["ostats_suture"], axis=1), np.argmax(output["ostats_suture"], axis=1), average='macro')
    metrics["ostats_flow_F1"] = f1_score(np.argmax(target["ostats_flow"], axis=1), np.argmax(output["ostats_flow"], axis=1), average='macro')
    metrics["ostats_knowledge_F1"] = f1_score(np.argmax(target["ostats_knowledge"], axis=1), np.argmax(output["ostats_knowledge"], axis=1), average='macro')
    metrics["ostats_performance_F1"] = f1_score(np.argmax(target["ostats_performance"], axis=1), np.argmax(output["ostats_performance"], axis=1), average='macro')
    return metrics

# %%
def train(epoch):
    for i, (coord,score,suture, group, time, cls, times_sutures, ostats_respect, ostats_motion, ostats_instrument, ostats_suture, ostats_flow, ostats_knowledge, ostats_performance, ostats_final_quality) in enumerate(train_loader):
        total_loss = 0
        coord, score, suture, group, time, cls, times_sutures = coord.to(device), score.to(device), suture.to(device), group.to(device), time.to(device), cls.to(device), times_sutures.to(device)
        ostats_respect, ostats_motion, ostats_instrument, ostats_suture, ostats_flow, ostats_knowledge, ostats_performance, ostats_final_quality = ostats_respect.to(device), ostats_motion.to(device), ostats_instrument.to(device), ostats_suture.to(device), ostats_flow.to(device), ostats_knowledge.to(device), ostats_performance.to(device), ostats_final_quality.to(device)
        target = {"score": score, "suture": suture, "group": group, "time": time, "class": cls, "time_reg": times_sutures, "ostats_respect": ostats_respect, "ostats_motion": ostats_motion, "ostats_instrument": ostats_instrument, "ostats_suture": ostats_suture, "ostats_flow": ostats_flow, "ostats_knowledge": ostats_knowledge, "ostats_performance": ostats_performance, "ostats_final_quality": ostats_final_quality}
        with torch.cuda.amp.autocast():
            output = model(coord)
            loss = loss_function(output, target, epoch=epoch)
            loss += ostats_loss(output, target)
            if epoch > CFG.epoch * 4 // 5:
                if CFG.task == 1:
                    loss -= ostats_loss(output, target)
                else:
                    loss -= loss_function(output, target, epoch=epoch)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    total_loss /= i
    wandb.log({"train_loss": total_loss})
    print(total_loss)

# %%
def valid(epoch):
    for i, (coord,score,suture, group, time, cls, times_sutures, ostats_respect, ostats_motion, ostats_instrument, ostats_suture, ostats_flow, ostats_knowledge, ostats_performance, ostats_final_quality) in tqdm(enumerate(valid_loader)):
        total_loss = 0
        target = {"score": score, "suture": suture, "group": group, "time": time, "class": cls, "time_reg": times_sutures, "ostats_respect": ostats_respect, "ostats_motion": ostats_motion, "ostats_instrument": ostats_instrument, "ostats_suture": ostats_suture, "ostats_flow": ostats_flow, "ostats_knowledge": ostats_knowledge, "ostats_performance": ostats_performance, "ostats_final_quality": ostats_final_quality}
        coord = coord.to(device)
        with torch.cuda.amp.autocast():
            output = model(coord)
            output = {key: value.cpu().detach().numpy() for key, value in output.items()}
        if i == 0:
            final_pred = output
            final_target = target
        else:
            for key, value in output.items():
                final_pred[key] = np.concatenate([final_pred[key], value], axis=0)
            for key, value in target.items():
                final_target[key] = np.concatenate([final_target[key], value], axis=0)
        
    metrics = calc_metrics(final_pred, final_target)
    print(f"epoch {epoch}:")
    for key, value in metrics.items():
        print(f"{key} \t : {value:.4f}")
        wandb.log({key: value})
    if CFG.task == 1: 
        return metrics["F1"]
    else:
        scr = metrics["ostats_respect_F1"] +  metrics["ostats_motion_F1"] + metrics["ostats_instrument_F1"] + metrics["ostats_suture_F1"] + metrics["ostats_flow_F1"] + metrics["ostats_knowledge_F1"] + metrics["ostats_performance_F1"]
        return scr / 7

# %%
best = 0
for epoch in range(CFG.epoch):
    train(epoch)
    score = valid(epoch)
    if best < score:
        best = score
        torch.save(model.state_dict(), CFG.model_path)
    scheduler.step()

# %%


# %%


# %%




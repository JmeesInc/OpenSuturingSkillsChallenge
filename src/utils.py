import numpy as np
import torch
import torch.nn as nn


def augment_sequence(sequence, target_length=2400):
    seq_len = sequence.shape[0]
    step_size = seq_len // target_length
    indices = np.arange(0, seq_len, step_size)
    random_offsets = np.random.randint(0, step_size, size=len(indices))
    selected_indices = indices + random_offsets
    selected_indices = np.sort(selected_indices[:target_length])
    augmented_sequence = sequence[selected_indices]
    
    return augmented_sequence


class OSSTestDataset(torch.utils.data.Dataset):
    def __init__(self, paths_to_coords):
        self.paths = paths_to_coords
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        aux = []
        coord = self.paths[idx]
        coord = np.load(coord).astype(np.float32)
        coord[:, :, 0] /= 1980
        coord[:, :, 1] /= 1080
        coord = coord.reshape(-1, 8)
        coord = augment_sequence(coord)
        return coord

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
        self.conv4 = nn.Sequential(nn.Conv1d(2400 // 15, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
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
        group = self.group_head(out).sigmoid()
        time = self.time_head(out).sigmoid()
        cls = self.classifier(out).sigmoid()

        ostats_respect = self.ostats_respect(out).sigmoid()
        ostats_motion = self.ostats_motion(out).sigmoid()
        ostats_instrument = self.ostats_instrument(out).sigmoid()
        ostats_suture = self.ostats_suture(out).sigmoid()
        ostats_flow = self.ostats_flow(out).sigmoid()
        ostats_knowledge = self.ostats_knowledge(out).sigmoid()
        ostats_performance = self.ostats_performance(out).sigmoid()

        return {"score": score, "suture": suture, "group": group, "time": time, "class": cls, "time_reg": time_reg, "ostats_respect": ostats_respect, "ostats_motion": ostats_motion, "ostats_instrument": ostats_instrument, "ostats_suture": ostats_suture, "ostats_flow": ostats_flow, "ostats_knowledge": ostats_knowledge, "ostats_performance": ostats_performance}

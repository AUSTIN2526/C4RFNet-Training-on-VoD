# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, in_ch, out_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 2048, 1)
        self.bn3 = nn.BatchNorm1d(2048)

        self.conv_fuse1 = nn.Conv1d(64 + 2048, 2048, 1)
        self.bn_fuse1 = nn.BatchNorm1d(2048)
        self.conv_fuse2 = nn.Conv1d(2048, out_dim, 1)
        self.bn_fuse2 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x: (B, N, C)
        x = x.transpose(1, 2)                 # (B, C, N)
        h1 = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        h2 = F.relu(self.bn2(self.conv2(h1))) # (B, 128, N)
        h3 = F.relu(self.bn3(self.conv3(h2))) # (B, 2048, N)
        g = torch.max(h3, dim=2, keepdim=True)[0]  # (B, 2048, 1)
        g_rep = g.repeat(1, 1, h1.shape[2])        # (B, 2048, N)
        fused = torch.cat([h1, g_rep], dim=1)      # (B, 2112, N)
        fused = F.relu(self.bn_fuse1(self.conv_fuse1(fused)))  # (B, 2048, N)
        fused = F.relu(self.bn_fuse2(self.conv_fuse2(fused)))  # (B, 1024, N)
        out = fused.max(dim=2)[0]                  # (B, 1024)
        return out

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class FoldingDecoder(nn.Module):
    def __init__(self, latent_dim=2048, grid_h=16, grid_w=32):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.latent_dim = latent_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(2 + latent_dim, 2051),
            nn.ReLU(inplace=True),
            nn.Linear(2051, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        self.refine = nn.Sequential(
            nn.Linear(3 + latent_dim, 2051),
            nn.ReLU(inplace=True),
            nn.Linear(2051, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def _build_grid(self, b, device):
        u = torch.linspace(-1, 1, self.grid_h, device=device)
        v = torch.linspace(-1, 1, self.grid_w, device=device)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        grid = torch.stack([uu, vv], dim=-1).view(-1, 2)
        return grid.unsqueeze(0).repeat(b, 1, 1)

    def forward(self, global_feat):
        B = global_feat.size(0)
        device = global_feat.device
        grid = self._build_grid(B, device)                # (B,M,2)
        M = grid.size(1)
        g = global_feat.unsqueeze(1).expand(B, M, self.latent_dim)
        x1_in = torch.cat([grid, g], dim=-1)              # (B,M,2+F)
        xyz = self.mlp1(x1_in)                            # (B,M,3)
        x2_in = torch.cat([xyz, g], dim=-1)               # (B,M,3+F)
        xyz_delta = self.refine(x2_in)                    # (B,M,3)
        return xyz + xyz_delta                            # (B,M,3)

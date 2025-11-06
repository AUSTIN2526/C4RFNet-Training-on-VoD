# -*- coding: utf-8 -*-
import torch
from .losses import chamfer_distance, local_topk_loss
from .pointnet_encoder import PointNetEncoder
from .folding_decoder import FoldingDecoder

class C4RFNet(torch.nn.Module):
    def __init__(self, enc_dim=1024, grid_h=16, grid_w=32, local_k=8,
                 loss_w_global=1.0, loss_w_local=1.0):
        super().__init__()
        self.radar_in_ch = 6
        self.image_in_ch = 5
        self.enc_dim = enc_dim
        self.fusion_dim = enc_dim * 2
        self.local_k = local_k
        self.loss_w_global = loss_w_global
        self.loss_w_local = loss_w_local

        self.enc_radar = PointNetEncoder(self.radar_in_ch, self.enc_dim)
        self.enc_image = PointNetEncoder(self.image_in_ch, self.enc_dim)
        self.dec = FoldingDecoder(self.fusion_dim, grid_h, grid_w)

    def forward(self, radar_pts, img_pseudo_pts):
        gr = self.enc_radar(radar_pts)               # (B,1024)
        gi = self.enc_image(img_pseudo_pts)          # (B,1024)
        fused = torch.concat([gr, gi], dim=-1)       # (B,2048)
        recon = self.dec(fused)                      # (B,M,3)
        return recon, fused

    def loss(self, recon, lidar_gt):
        loss_g = chamfer_distance(recon, lidar_gt)
        loss_l = local_topk_loss(recon, lidar_gt, k=self.local_k)
        return (self.loss_w_global * loss_g + self.loss_w_local * loss_l).mean()

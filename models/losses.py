# -*- coding: utf-8 -*-
import torch

def pairwise_sq_dist(x, y):
    # x: (B, Nx, C), y: (B, Ny, C)
    x2 = (x ** 2).sum(dim=-1, keepdim=True)
    y2 = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)
    xy = x @ y.transpose(1, 2)
    return x2 + y2 - 2 * xy

def chamfer_distance(p1, p2):
    # p1: (B, N1, 3), p2: (B, N2, 3)
    d2 = pairwise_sq_dist(p1, p2)
    min1, _ = d2.min(dim=2)  # (B, N1)
    min2, _ = d2.min(dim=1)  # (B, N2)
    return min1.mean(dim=1) + min2.mean(dim=1)

def local_topk_loss(p1, p2, k=8):
    # p1: (B, N1, 3), p2: (B, N2, 3)
    d = pairwise_sq_dist(p1, p2)
    d_T = d.transpose(1, 2).contiguous()
    k1 = min(k, p2.shape[1])
    k2 = min(k, p1.shape[1])
    v1, _ = torch.topk(d, k=k1, dim=2, largest=False)
    v2, _ = torch.topk(d_T, k=k2, dim=2, largest=False)
    return v1.mean(dim=(1, 2)) + v2.mean(dim=(1, 2))

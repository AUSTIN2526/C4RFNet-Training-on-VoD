# -*- coding: utf-8 -*-
import torch

def pairwise_sq_dist(x, y):
    """
    計算 pairwise squared distance。
    x: (B, Nx, C)
    y: (B, Ny, C)
    回傳: (B, Nx, Ny)
    """
    x2 = (x ** 2).sum(dim=-1, keepdim=True)              # (B, Nx, 1)
    y2 = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, Ny)
    xy = x @ y.transpose(1, 2)                           # (B, Nx, Ny)
    return x2 + y2 - 2 * xy


def chamfer_distance(p1, p2):
    """
    Loss_global: Squared L2 distance
    """
    d2 = pairwise_sq_dist(p1, p2)
    min1, _ = d2.min(dim=2)
    min2, _ = d2.min(dim=1)
    return min1.mean(dim=1) + min2.mean(dim=1)


def local_topk_loss(p1, p2, k=8):
    """
    Loss_local:
    修正：移除了 torch.sqrt。
    因為 Global Loss 使用的是 Squared L2，這裡也應該使用 Squared distance 來保持量級一致，
    否則 Linear distance 在誤差 < 1 時會比 Squared distance 大很多，導致權重失衡。
    """
    # 使用 squared distance，不要開根號
    d2 = pairwise_sq_dist(p1, p2)                             # (B, N1, N2)
    
    # 對稱性計算
    d2_T = d2.transpose(1, 2).contiguous()                    # (B, N2, N1)

    k1 = min(k, p2.shape[1])
    k2 = min(k, p1.shape[1])

    # 每個點找 K 個最近鄰 (Squared distance 最小即 Euclidean distance 最小)
    v1, _ = torch.topk(d2, k=k1, dim=2, largest=False)        # (B, N1, K1)
    v2, _ = torch.topk(d2_T, k=k2, dim=2, largest=False)      # (B, N2, K2)

    return v1.mean(dim=(1, 2)) + v2.mean(dim=(1, 2))
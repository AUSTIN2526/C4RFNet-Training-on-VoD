# -*- coding: utf-8 -*-
import torch

def pairwise_sq_dist(x, y):
    """
    計算兩批點雲的 pairwise squared distance。
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
    論文式(20)的 Loss_global，使用 squared L2 distance：
        Loss_global(P1, P2) =
            1/|P1| sum_{x in P1} min_{y in P2} ||x - y||_2^2 +
            1/|P2| sum_{y in P2} min_{x in P1} ||y - x||_2^2
    p1: (B, N1, 3)
    p2: (B, N2, 3)
    回傳: (B,) 每個 batch 的 loss 值
    """
    d2 = pairwise_sq_dist(p1, p2)          # squared distance
    min1, _ = d2.min(dim=2)                # (B, N1)
    min2, _ = d2.min(dim=1)                # (B, N2)
    return min1.mean(dim=1) + min2.mean(dim=1)


def local_topk_loss(p1, p2, k=8):
    """
    論文式(21)~(23)的 Loss_local：
    - 先建距離矩陣 D，元素為歐氏距離 (有開根號)。
    - 針對每一列/每一行取 TopK 最小距離，再取平均。

    p1: (B, N1, 3)
    p2: (B, N2, 3)
    回傳: (B,) 每個 batch 的 local loss
    """
    # 先算 squared distance
    d2 = pairwise_sq_dist(p1, p2)                             # (B, N1, N2)
    # 再轉成歐氏距離 (論文中的 D_ij 定義)
    d = torch.sqrt(torch.clamp(d2, min=1e-9))                 # (B, N1, N2)

    d_T = d.transpose(1, 2).contiguous()                      # (B, N2, N1)

    k1 = min(k, p2.shape[1])
    k2 = min(k, p1.shape[1])

    # 對 P1: 每個點對 P2 的 K 個最近鄰距離
    v1, _ = torch.topk(d, k=k1, dim=2, largest=False)         # (B, N1, K1)
    # 對 P2: 每個點對 P1 的 K 個最近鄰距離
    v2, _ = torch.topk(d_T, k=k2, dim=2, largest=False)       # (B, N2, K2)

    # 論文式(23) 取平均
    return v1.mean(dim=(1, 2)) + v2.mean(dim=(1, 2))

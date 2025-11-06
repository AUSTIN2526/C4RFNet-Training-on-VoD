# -*- coding: utf-8 -*-
import os
import random
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.c4rfnet import C4RFNet
from data.vod import VoDDataset, build_id_list, split_train_val
from trainer import Trainer


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For CUDA determinism (PyTorch >= 1.8); may reduce performance
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # cuDNN/algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Ensure each worker has a deterministic seed
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train C4RFNet on VoD with deterministic seeding and CLI args.")
    # Data
    p.add_argument("--base_dir", type=str, default="VoD", help="Dataset root directory.")
    p.add_argument("--down_h", type=int, default=19, help="Downsampled height.")
    p.add_argument("--down_w", type=int, default=30, help="Downsampled width.")
    p.add_argument("--radar_points", type=int, default=512)
    p.add_argument("--lidar_points", type=int, default=1024)
    p.add_argument("--val_ratio", type=float, default=0.1)
    # Loader
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--drop_last_train", action="store_true", default=True)
    p.add_argument("--no-drop_last_train", dest="drop_last_train", action="store_false")
    p.add_argument("--drop_last_val", action="store_true", default=False)
    p.add_argument("--no-drop_last_val", dest="drop_last_val", action="store_false")
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--no-pin_memory", dest="pin_memory", action="store_false")
    # Model
    p.add_argument("--enc_dim", type=int, default=1024)
    p.add_argument("--grid_h", type=int, default=16)
    p.add_argument("--grid_w", type=int, default=32)
    p.add_argument("--local_k", type=int, default=8)
    p.add_argument("--loss_w_global", type=float, default=1.0)
    p.add_argument("--loss_w_local", type=float, default=1.0)
    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--t_max", type=int, default=40, help="CosineAnnealingLR T_max.")
    p.add_argument("--eta_min", type=float, default=1e-5, help="CosineAnnealingLR eta_min.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stopping", type=int, default=5)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints_c4rfnet")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    base_dir = args.base_dir
    ids = build_id_list(base_dir)
    if len(ids) == 0:
        raise RuntimeError("找不到資料。請確認 VOD 目錄結構與檔案存在。")

    train_ids, val_ids = split_train_val(ids, val_ratio=args.val_ratio, seed=args.seed)

    train_set = VoDDataset(
        base_dir,
        train_ids,
        down_hw=(args.down_h, args.down_w),
        radar_points=args.radar_points,
        lidar_points=args.lidar_points,
    )
    val_set = VoDDataset(
        base_dir,
        val_ids,
        down_hw=(args.down_h, args.down_w),
        radar_points=args.radar_points,
        lidar_points=args.lidar_points,
    )

    # Deterministic shuffling via generator
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last_train,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last_val,
        worker_init_fn=seed_worker,
        generator=g,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C4RFNet(
        enc_dim=args.enc_dim,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        local_k=args.local_k,
        loss_w_global=args.loss_w_global,
        loss_w_local=args.loss_w_local,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        early_stopping=args.early_stopping,
        save_dir=args.save_dir,
    )

    trainer.train(show_loss=True)


if __name__ == "__main__":
    main()

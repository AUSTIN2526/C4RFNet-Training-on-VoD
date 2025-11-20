# -*- coding: utf-8 -*-
import os
import random
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from PIL import Image

# 假設這些模組都在您的目錄結構中
from models.c4rfnet import C4RFNet
from preprocess.vod import (
    VoDDataset,
    build_id_list,
    split_train_val,
    load_vod_calib,
    image_to_pseudo_pointcloud,
    load_radar_points,
)
from trainer import Trainer


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    p = argparse.ArgumentParser(description="C4RFNet on VoD")
    p.add_argument("--mode", type=str, default="train", choices=["train", "export"])
    p.add_argument("--base_dir", type=str, default="view_of_delft_PUBLIC")
    p.add_argument("--down_h", type=int, default=19)
    p.add_argument("--down_w", type=int, default=30)
    p.add_argument("--radar_points", type=int, default=512)
    p.add_argument("--lidar_points", type=int, default=1024)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--enc_dim", type=int, default=1024)
    p.add_argument("--grid_h", type=int, default=16)
    p.add_argument("--grid_w", type=int, default=32)
    p.add_argument("--local_k", type=int, default=8)
    p.add_argument("--loss_w_global", type=float, default=1.0)
    p.add_argument("--loss_w_local", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--t_max", type=int, default=160)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stopping", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints_c4rfnet")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--export_dir", type=str, default="./enhance_result")
    p.add_argument("--max_export", type=int, default=-1)
    return p.parse_args()


def cam_to_sensor(xyz_cam, E):
    """相機座標系 -> Radar/LiDAR座標系"""
    n = xyz_cam.shape[0]
    xyz_h = np.concatenate([xyz_cam, np.ones((n, 1))], axis=1)
    E_inv = np.linalg.inv(E)
    sensor_h = (E_inv @ xyz_h.T).T
    return sensor_h[:, :3].astype(np.float32)


def export_enhance_bin_robust(args):
    base_dir = args.base_dir
    ids = build_id_list(base_dir)
    if len(ids) == 0:
        raise RuntimeError("No data found.")

    os.makedirs(args.export_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = C4RFNet(
        enc_dim=args.enc_dim,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        local_k=args.local_k,
        loss_w_global=args.loss_w_global,
        loss_w_local=args.loss_w_local,
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(args.save_dir, "best_model.ckpt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f"Exporting to {args.export_dir} using {ckpt_path}...")
    down_hw = (args.down_h, args.down_w)
    exported = 0

    with torch.no_grad():
        for fid in ids:
            if args.max_export > 0 and exported >= args.max_export:
                break

            radar_file = os.path.join(base_dir, "radar/training/velodyne", f"{fid}.bin")
            radar_calib_file = os.path.join(base_dir, "radar/training/calib", f"{fid}.txt")
            img_path = os.path.join(base_dir, "lidar/training/image_2", f"{fid}.jpg")

            if not all(os.path.isfile(f) for f in [radar_file, radar_calib_file, img_path]):
                continue

            # 1. 讀取完整原始 Radar (保留所有點與屬性)
            raw_radar_full = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)

            # 2. 準備網路輸入
            E_radar, K = load_vod_calib(radar_calib_file)
            img = Image.open(img_path)
            W, H = img.size
            pseudo = image_to_pseudo_pointcloud(img_path, down_hw)
            
            # 注意：這裡的 load_radar_points 僅用於採樣生成 input，不影響原始 raw_radar_full
            radar_net_input = load_radar_points(
                radar_file, E_radar, K, (W, H), target_num_points=args.radar_points
            )
            radar_t = torch.from_numpy(radar_net_input).float().unsqueeze(0).to(device)
            pseudo_t = torch.from_numpy(pseudo).float().unsqueeze(0).to(device)

            # 3. 推論
            recon_cam, _ = model(radar_t, pseudo_t)
            recon_cam = recon_cam[0].cpu().numpy().astype(np.float32)

            # 4. 轉回雷達座標系
            recon_xyz_sensor = cam_to_sensor(recon_cam, E_radar)

            # 5. 補特徵 [RCS, vr, vr_comp, time_id]
            # FIX: RCS 補非零值 (如 1.0)，避免被下游檢測器當作雜訊過濾
            num_recon = recon_xyz_sensor.shape[0]
            recon_feats = np.zeros((num_recon, 4), dtype=np.float32)
            recon_feats[:, 0] = 1.0  # Set dummy RCS to 1.0
            
            recon_full_7d = np.hstack([recon_xyz_sensor, recon_feats])

            # 6. 合併並存檔
            final_output = np.concatenate([raw_radar_full, recon_full_7d], axis=0).astype(np.float32)
            final_output.tofile(os.path.join(args.export_dir, f"{fid}.bin"))
            exported += 1
            if exported % 100 == 0:
                print(f"Exported {exported}")

    print(f"Done. Total {exported} frames.")

def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.mode == "export":
        export_enhance_bin_robust(args)
        return

    # Training Mode
    base_dir = args.base_dir
    ids = build_id_list(base_dir)
    if len(ids) == 0:
        raise RuntimeError("No data found.")

    train_ids, val_ids = split_train_val(ids, val_ratio=args.val_ratio, seed=args.seed)
    
    train_set = VoDDataset(base_dir, train_ids, (args.down_h, args.down_w),
                           args.radar_points, args.lidar_points)
    val_set = VoDDataset(base_dir, val_ids, (args.down_h, args.down_w),
                         args.radar_points, args.lidar_points)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=False, worker_init_fn=seed_worker, generator=g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C4RFNet(args.enc_dim, args.grid_h, args.grid_w, args.local_k,
                    args.loss_w_global, args.loss_w_local).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    
    trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader,
                      device=device, epochs=args.epochs, grad_clip=args.grad_clip,
                      early_stopping=args.early_stopping, save_dir=args.save_dir)
    
    trainer.train()

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
C4RFNet Complete Implementation - Improved Version
修正內容：
1. 導出模式 (Export) 現在會保留原始 Radar 的完整數據 (含 time_id 與 FoV 外的點)。
2. 生成的點雲會被轉換回 Radar 座標系並附加在原始數據之後。
"""
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
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    p = argparse.ArgumentParser(
        description="C4RFNet on VoD: train or export enhanced 4D-Radar BIN."
    )
    # Mode
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "export"],
        help="train: 訓練 C4RFNet；export: 載入訓練好的模型輸出增強後的 BIN。",
    )

    # Data
    p.add_argument("--base_dir", type=str, default="view_of_delft_PUBLIC", help="Dataset root directory.")
    p.add_argument("--down_h", type=int, default=19, help="Downsampled height.")
    p.add_argument("--down_w", type=int, default=30, help="Downsampled width.")
    p.add_argument("--radar_points", type=int, default=512)
    p.add_argument("--lidar_points", type=int, default=1024)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # Loader (for train)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
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

    # Optim (for train)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--t_max", type=int, default=160, help="CosineAnnealingLR T_max.")
    p.add_argument("--eta_min", type=float, default=1e-5, help="CosineAnnealingLR eta_min.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stopping", type=int, default=10)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints_c4rfnet")

    # Export-related
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="導出時使用的模型權重路徑；預設為 <save_dir>/best_model.ckpt。",
    )
    p.add_argument(
        "--export_dir",
        type=str,
        default="./enhance_result",
        help="輸出增強後點雲 BIN 的目錄。",
    )
    p.add_argument(
        "--max_export",
        type=int,
        default=-1,
        help="最多導出多少 frame；-1 代表全部。",
    )

    return p.parse_args()


def cam_to_sensor(xyz_cam, E):
    """相機座標系 -> Radar/LiDAR座標系"""
    n = xyz_cam.shape[0]
    xyz_h = np.concatenate([xyz_cam, np.ones((n, 1))], axis=1)
    E_inv = np.linalg.inv(E)
    sensor_h = (E_inv @ xyz_h.T).T
    return sensor_h[:, :3].astype(np.float32)


def export_enhance_bin_robust(args):
    """
    改良版導出函數：
    1. 保留完整的原始 Radar 數據 (包含 time_id 與 FoV 外的點)。
    2. 僅將網絡生成的點 (Reconstructed) 轉換回 Radar 座標系並附加在後面。
    3. 生成點的非幾何特徵 (RCS, v_r, v_r_comp, time_id) 補 0。
    """
    base_dir = args.base_dir
    ids = build_id_list(base_dir)

    if len(ids) == 0:
        raise RuntimeError("找不到數據")

    os.makedirs(args.export_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    model = C4RFNet(
        enc_dim=args.enc_dim,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        local_k=args.local_k,
        loss_w_global=args.loss_w_global,
        loss_w_local=args.loss_w_local,
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(args.save_dir, "best_model.ckpt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\n{'='*60}")
    print(f"C4RFNet Export - Robust Version (保留原始完整數據)")
    print(f"{'='*60}")
    print(f"使用權重: {ckpt_path}")
    print(f"輸出目錄: {args.export_dir}")
    print(f"{'='*60}\n")

    exported = 0
    down_hw = (args.down_h, args.down_w)

    with torch.no_grad():
        for fid in ids:
            if args.max_export > 0 and exported >= args.max_export:
                break

            # === 1. 準備檔案路徑 ===
            radar_file = os.path.join(base_dir, "radar/training/velodyne", f"{fid}.bin")
            radar_calib_file = os.path.join(base_dir, "radar/training/calib", f"{fid}.txt")
            img_path = os.path.join(base_dir, "lidar/training/image_2", f"{fid}.jpg")

            if not all(os.path.isfile(f) for f in [radar_file, radar_calib_file, img_path]):
                print(f"[SKIP] {fid}: 資料缺失")
                continue

            # === 2. 讀取數據 ===
            # 2.1 讀取【完整】的原始 Radar 數據 (N_raw, 7) -> [x, y, z, RCS, vr, vr_comp, time_id]
            # 用於最後寫入檔案，確保數據不遺失
            raw_radar_full = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)

            # 2.2 讀取用於【模型輸入】的數據 (必須符合訓練時的 Preprocessing: FoV Crop + Sampling)
            # 這裡我們再次使用 vod.py 的 load_radar_points，只為了給網路吃
            E_radar, K = load_vod_calib(radar_calib_file)
            img = Image.open(img_path)
            W, H = img.size

            # pseudo point cloud (Ni, 5)
            pseudo = image_to_pseudo_pointcloud(img_path, down_hw)
            
            # radar input for network (1, 512, 6) - 僅用於推論
            radar_net_input = load_radar_points(
                radar_file, E_radar, K, (W, H),
                target_num_points=args.radar_points
            )
            # 分離特徵 (xyz 與 features) 雖然 load_radar_points 已經混合了，這裡為了轉 tensor
            radar_t = torch.from_numpy(radar_net_input).float().unsqueeze(0).to(device)
            pseudo_t = torch.from_numpy(pseudo).float().unsqueeze(0).to(device)

            # === 3. 網路推論 ===
            # recon_cam: (1, M, 3) 在相機座標系
            recon_cam, _ = model(radar_t, pseudo_t)
            recon_cam = recon_cam[0].cpu().numpy().astype(np.float32)  # (M, 3)

            # === 4. 後處理 ===
            # [cite_start]4.1 將生成的點轉回 Radar 座標系 [cite: 17, 271]
            # 論文式 (17) 的逆變換
            recon_xyz_sensor = cam_to_sensor(recon_cam, E_radar)

            # 4.2 為生成點補上特徵，使其變成 7 維格式
            # 原始格式: [x, y, z, RCS, vr, vr_comp, time_id]
            # 生成點只有 xyz，其餘補 0
            num_recon = recon_xyz_sensor.shape[0]
            recon_feats = np.zeros((num_recon, 4), dtype=np.float32) # [RCS, vr, vr_comp, time_id]
            recon_full_7d = np.hstack([recon_xyz_sensor, recon_feats])

            # === 5. 合併輸出 ===
            # 最終結果 = 完整原始 Radar + 生成的 Radar
            # 這樣既保留了原始資訊 (包含 time_id 和 FoV 外的點)，又加上了增強點
            final_output = np.concatenate([raw_radar_full, recon_full_7d], axis=0).astype(np.float32)

            # === 6. 寫入檔案 ===
            out_path = os.path.join(args.export_dir, f"{fid}.bin")
            final_output.tofile(out_path)

            exported += 1
            if exported % 50 == 0:
                print(f"已導出 {exported} 個 frame")

    print(f"導出完成！總計 {exported} 個檔案。")


def main():
    """
    主函數：根據 mode 參數選擇訓練或導出
    """
    args = parse_args()
    seed_everything(args.seed)

    # === 導出模式 ===
    if args.mode == "export":
        print("\n" + "="*60)
        print("模式：導出增強點雲")
        print("="*60)
        export_enhance_bin_robust(args)
        return

    # === 訓練模式 (保持不變) ===
    print("\n" + "="*60)
    print("模式：訓練 C4RFNet")
    print("="*60)

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

    print(f"\n開始訓練...")
    print(f"  訓練集: {len(train_ids)} frames")
    print(f"  驗證集: {len(val_ids)} frames")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {device}")
    print(f"  模型保存路徑: {args.save_dir}\n")

    trainer.train(show_loss=True)

    print("\n訓練完成！")
    print(f"最佳模型已保存至: {os.path.join(args.save_dir, 'best_model.ckpt')}")
    print(f"\n提示：使用以下命令導出增強點雲：")
    print(f"  python main.py --mode export --checkpoint {os.path.join(args.save_dir, 'best_model.ckpt')}")


if __name__ == "__main__":
    main()
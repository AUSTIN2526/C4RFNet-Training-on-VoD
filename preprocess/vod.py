# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_vod_calib(calib_file):
    """
    讀取 VoD / KITTI 風格的 calib 檔，回傳：
    - E : (4,4) 由感測器座標系 (velodyne/radar) -> 相機座標系 的外參
    - K : (3,3) 相機內參 (由 P2 取前 3x3)
    """
    calib = {}
    with open(calib_file, "r") as f:
        for line in f.readlines():
            if ":" in line:
                key, value = line.split(":", 1)
                calib[key.strip()] = np.array(
                    [float(x) for x in value.split()], dtype=np.float32
                )

    # Tr_velo_to_cam: 3x4，代表「該感測器座標系 -> 相機座標系」
    Tr = calib["Tr_velo_to_cam"].reshape(3, 4).astype(np.float32)
    E = np.vstack([Tr, [0, 0, 0, 1]]).astype(np.float32)

    # P2: 3x4，取前 3x3 當 K
    K = calib["P2"].reshape(3, 4)[:, :3].astype(np.float32)
    return E, K


def image_to_pseudo_pointcloud(img_path, downsample_hw=(19, 30)):
    """
    依論文式 (19) 建立 pseudo point cloud：
    [i, j, R, G, B]
    
    修正：
    原本代碼直接使用絕對像素座標 i, j (例如 0~19)，這會導致數值範圍與 RGB (0~1) 差異過大，
    且不符合 PointNet 對輸入座標 [-1, 1] 的假設。
    此處已修正為將 (i, j) 正規化至 [-1, 1]。
    """
    img = Image.open(img_path).convert("RGB")
    H_d, W_d = downsample_hw
    img_resized = img.resize((W_d, H_d), Image.BILINEAR)
    arr = np.array(img_resized)  # (H_d, W_d, 3)

    h, w, _ = arr.shape
    pseudo = np.zeros((h * w, 5), dtype=np.float32)
    idx = 0
    for i in range(h):
        for j in range(w):
            r, g, b = arr[i, j]
            # === FIX START: Normalize coordinates to [-1, 1] ===
            norm_i = (i / (h - 1)) * 2 - 1 if h > 1 else 0
            norm_j = (j / (w - 1)) * 2 - 1 if w > 1 else 0
            # === FIX END ===
            
            pseudo[idx] = [norm_i, norm_j, r / 255.0, g / 255.0, b / 255.0]
            idx += 1
    return pseudo  # (H_d*W_d, 5)


def project_and_filter(xyz_cam, K, img_size):
    """
    將 3D 點 (相機座標系) 透過內參 K 投影到影像平面，
    並依照影像大小 (W,H) 做 FoV 篩選。
    """
    # xyz_cam: (N,3)
    Pi = (K @ xyz_cam.T).T  # (N,3)
    zc = Pi[:, 2]
    uv = Pi[:, :2] / (zc[:, None] + 1e-8)

    W, H = img_size
    mask = (
        (zc > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < W)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < H)
    )
    return mask


def load_radar_points(radar_file, E_radar, K, img_size, target_num_points=512):
    """
    讀取 VoD 4D Radar 點雲並轉換至相機座標系。
    回傳: (N, 6) [x, y, z, RCS, v_r, v_r_comp]
    """
    data = np.fromfile(radar_file, dtype=np.float32)
    if data.size == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32)

    pts = data.reshape(-1, 7)[:, :6]  # x,y,z,RCS,v_r,v_r_comp
    xyz = pts[:, :3].astype(np.float32)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)

    # 雷達座標 -> 相機座標
    Pc_h = (E_radar @ np.hstack([xyz, ones]).T).T  # (N,4)
    Pc = Pc_h[:, :3]

    # 只保留在相機視野內的點
    mask = project_and_filter(Pc, K, img_size)
    valid_xyz = Pc[mask]
    valid_feat = pts[mask, 3:].astype(np.float32)

    radar_cam = np.hstack([valid_xyz, valid_feat]).astype(np.float32)  # (N,6)

    # Sampling
    n = len(radar_cam)
    if n == 0:
        radar_cam = np.zeros((target_num_points, 6), dtype=np.float32)
    elif n < target_num_points:
        idx = np.random.choice(n, target_num_points - n, replace=True)
        radar_cam = np.vstack([radar_cam, radar_cam[idx]])
    elif n > target_num_points:
        idx = np.random.choice(n, target_num_points, replace=False)
        radar_cam = radar_cam[idx]
    return radar_cam


def load_lidar_label(lidar_file, E_lidar, K, img_size, target_label_points=1024):
    """
    讀取 VoD LiDAR 點雲並轉換至相機座標系。
    回傳: (N, 3) [x, y, z]
    """
    data = np.fromfile(lidar_file, dtype=np.float32)
    if data.size == 0:
        return np.zeros((target_label_points, 3), dtype=np.float32)

    pts = data.reshape(-1, 4)[:, :3].astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)

    # LiDAR 座標 -> 相機座標
    Pc_h = (E_lidar @ np.hstack([pts, ones]).T).T  # (N,4)
    Pc = Pc_h[:, :3]

    mask = project_and_filter(Pc, K, img_size)
    lidar_cam = Pc[mask].astype(np.float32)

    n = len(lidar_cam)
    if n == 0:
        lidar_cam = np.zeros((target_label_points, 3), dtype=np.float32)
    elif n < target_label_points:
        idx = np.random.choice(n, target_label_points - n, replace=True)
        lidar_cam = np.vstack([lidar_cam, lidar_cam[idx]])
    elif n > target_label_points:
        idx = np.random.choice(n, target_label_points, replace=False)
        lidar_cam = lidar_cam[idx]
    return lidar_cam


class VoDDataset(Dataset):
    def __init__(self, base_dir, id_list, down_hw=(19, 30),
                 radar_points=512, lidar_points=1024):
        self.base = base_dir
        self.ids = id_list
        self.down_hw = down_hw
        self.radar_points = radar_points
        self.lidar_points = lidar_points

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        radar_file = os.path.join(self.base, "radar/training/velodyne", f"{fid}.bin")
        lidar_file = os.path.join(self.base, "lidar/training/velodyne", f"{fid}.bin")
        radar_calib_file = os.path.join(self.base, "radar/training/calib", f"{fid}.txt")
        lidar_calib_file = os.path.join(self.base, "lidar/training/calib", f"{fid}.txt")
        img_path = os.path.join(self.base, "lidar/training/image_2", f"{fid}.jpg")

        E_radar, K = load_vod_calib(radar_calib_file)
        E_lidar, K2 = load_vod_calib(lidar_calib_file)

        with Image.open(img_path) as img:
            W, H = img.size

        pseudo = image_to_pseudo_pointcloud(img_path, self.down_hw)

        radar = load_radar_points(
            radar_file, E_radar, K, (W, H), target_num_points=self.radar_points
        )

        lidar = load_lidar_label(
            lidar_file, E_lidar, K2, (W, H), target_label_points=self.lidar_points
        )

        sample = {
            "radar": torch.from_numpy(radar).float(),
            "image": torch.from_numpy(pseudo).float(),
            "lidar": torch.from_numpy(lidar).float(),
            "E_radar": torch.from_numpy(E_radar).float(),
            "E_lidar": torch.from_numpy(E_lidar).float(),
            "fid": fid,
        }
        return sample


def build_id_list(base_dir, split="train"):
    split_file = os.path.join(base_dir, "radar", "ImageSets", f"{split}.txt")
    if os.path.isfile(split_file):
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f.readlines() if line.strip()]
        return ids

    bins = glob.glob(os.path.join(base_dir, "radar/training/velodyne", "*.bin"))
    ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in bins])
    return ids


def split_train_val(ids, val_ratio=0.1, seed=42):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val = max(1, int(n * val_ratio))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids
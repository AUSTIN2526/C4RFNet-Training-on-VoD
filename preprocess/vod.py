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
    其中 (i, j) 為下採樣影像上的像素座標，RGB 在這裡做 0~1 正規化。
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
            pseudo[idx] = [i, j, r / 255.0, g / 255.0, b / 255.0]
            idx += 1
    return pseudo  # (H_d*W_d, 5)


def project_and_filter(xyz_cam, K, img_size):
    """
    將 3D 點 (相機座標系) 透過內參 K 投影到影像平面，
    並依照影像大小 (W,H) 做 FoV 篩選。這對應論文中由 3D -> 影像座標的步驟。
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
    讀取 VoD 4D Radar 點雲：
    原始格式為 7 維 [x, y, z, RCS, v_r, v_r_comp, time_id]，
    論文只使用前 6 維 (忽略 time_id)。
    - 先由雷達座標系透過外參 E_radar 轉到相機座標系
    - 再用 K 投影到影像，做 FoV 篩選
    - 最後取/補成 target_num_points 個點
    
    回傳:
        radar_cam: (N, 6) 在相機座標系的點 [x,y,z,RCS,v_r,v_r_comp]
    """
    data = np.fromfile(radar_file, dtype=np.float32)
    if data.size == 0:
        # 防呆：空檔案
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

    # 按論文設定取 512 點 (Radar512)
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
    讀取 VoD LiDAR 點雲：
    原始格式為 4 維 [x, y, z, intensity]，只取前 3 維做幾何監督。
    - 用 LiDAR 自己的外參 E_lidar 轉到相機座標系
    - 用 K 投影、FoV 篩選
    - 取/補成 target_label_points (論文中的 Lidar1024)
    
    回傳:
        lidar_cam: (N, 3) 在相機座標系的點 [x,y,z]
    """
    data = np.fromfile(lidar_file, dtype=np.float32)
    if data.size == 0:
        return np.zeros((target_label_points, 3), dtype=np.float32)

    pts = data.reshape(-1, 4)[:, :3].astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)

    # LiDAR 座標 -> 相機座標（注意：這裡要用 LiDAR 的外參，不是 Radar 的）
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
    """
    專門給 C4RFNet 用的 VoD Dataset：
    - radar: 4D Radar512，在相機座標系，6 維 [x,y,z,RCS,v_r,v_r_comp]
    - image: pseudo point cloud，由影像下採樣後組成 (Ni,5)
    - lidar: Lidar1024，在相機座標系，3 維 [x,y,z]
    - E_radar: Radar 外參矩陣 (4,4)
    - E_lidar: LiDAR 外參矩陣 (4,4)
    """

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

        # 檔案路徑
        radar_file = os.path.join(
            self.base, "radar/training/velodyne", f"{fid}.bin"
        )
        lidar_file = os.path.join(
            self.base, "lidar/training/velodyne", f"{fid}.bin"
        )
        radar_calib_file = os.path.join(
            self.base, "radar/training/calib", f"{fid}.txt"
        )
        lidar_calib_file = os.path.join(
            self.base, "lidar/training/calib", f"{fid}.txt"
        )
        # 影像只需要一份（雷達與 LiDAR 共用同一顆相機）
        img_path = os.path.join(
            self.base, "lidar/training/image_2", f"{fid}.jpg"
        )

        # 讀取外參與內參：
        #   - Radar: E_radar, K
        #   - LiDAR: E_lidar, K2 (P2 相同，這裡只取內參，可忽略 K2 或再檢查一致性)
        E_radar, K = load_vod_calib(radar_calib_file)
        E_lidar, K2 = load_vod_calib(lidar_calib_file)

        # 影像大小
        with Image.open(img_path) as img:
            W, H = img.size

        # pseudo point cloud (Ni,5)
        pseudo = image_to_pseudo_pointcloud(img_path, self.down_hw)

        # Radar512 (在相機座標系下) (512,6)
        radar = load_radar_points(
            radar_file,
            E_radar,
            K,
            (W, H),
            target_num_points=self.radar_points,
        )

        # Lidar1024 ground truth (在相機座標系下) (1024,3)
        lidar = load_lidar_label(
            lidar_file,
            E_lidar,
            K2,
            (W, H),
            target_label_points=self.lidar_points,
        )

        sample = {
            "radar": torch.from_numpy(radar).float(),       # (512,6)
            "image": torch.from_numpy(pseudo).float(),      # (Ni,5)
            "lidar": torch.from_numpy(lidar).float(),       # (1024,3)
            "E_radar": torch.from_numpy(E_radar).float(),   # (4,4)
            "E_lidar": torch.from_numpy(E_lidar).float(),   # (4,4)
            "fid": fid,
        }
        return sample


def build_id_list(base_dir, split="train"):
    """
    依 VoD / KITTI 的 ImageSets 決定使用哪些 frame：
    - 優先讀取 radar/ImageSets/<split>.txt
    - 若不存在，退而求其次掃 radar/training/velodyne/*.bin
    """
    split_file = os.path.join(base_dir, "radar", "ImageSets", f"{split}.txt")
    if os.path.isfile(split_file):
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f.readlines() if line.strip()]
        return ids

    # fallback：無 ImageSets 時，直接列出 training/velodyne 的所有 .bin
    bins = glob.glob(os.path.join(base_dir, "radar/training/velodyne", "*.bin"))
    ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in bins])
    return ids


def split_train_val(ids, val_ratio=0.1, seed=42):
    """
    在指定的 id 列表上再做一次 train/val 拆分，
    方便交叉驗證或保留一部分做 validation。
    """
    ids = list(ids)  # 避免原地打亂呼叫者的 list
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val = max(1, int(n * val_ratio))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids





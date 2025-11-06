# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional

import torch
from torch.utils.data import DataLoader

from models.c4rfnet import C4RFNet
from data.vod import VoDDataset, build_id_list, split_train_val


def seed_everything(seed: int):
    import random
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
    import random
    random.seed(worker_seed)


@torch.no_grad()
def export_enhance1024(
    model_ckpt: str,
    base_dir: str = "VoD",
    out_root: Optional[str] = "output",
    batch_size: int = 16,
    num_workers: int = 0,
    split_ratio: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    seed_everything(seed)

    ids = build_id_list(base_dir)
    if len(ids) == 0:
        raise RuntimeError("No data found. Please check the VoD directory structure.")

    train_ids, val_ids = split_train_val(ids, val_ratio=split_ratio, seed=seed)

    def make_loader(id_list):
        ds = VoDDataset(
            base_dir,
            id_list,
            down_hw=(19, 30),
            radar_points=512,
            lidar_points=1024,
        )
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

    if not os.path.isfile(model_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {model_ckpt}")
    model = C4RFNet(
        enc_dim=1024,
        grid_h=32,
        grid_w=32,
        local_k=8,
        loss_w_global=1.0,
        loss_w_local=1.0,
    ).to(device)
    state = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if out_root is None:
        out_root = "output"
    os.makedirs(out_root, exist_ok=True)

    velodyne_dir = os.path.join(out_root, "velodyne")
    os.makedirs(velodyne_dir, exist_ok=True)

    def run_and_dump(loader):
        for batch in tqdm(loader, desc="Export Enhance1024"):
            radar = batch["radar"].to(device)
            image = batch["image"].to(device)
            fids = batch["fid"]
            recon, _ = model(radar, image)
            recon = recon.cpu().numpy().astype(np.float32)

            for b in range(recon.shape[0]):
                fid = fids[b]
                xyz = recon[b]
                inten = np.ones((xyz.shape[0], 1), dtype=np.float32)
                xyzi = np.hstack([xyz, inten])
                out_path = os.path.join(velodyne_dir, f"{fid}.bin")
                xyzi.tofile(out_path)

    run_and_dump(make_loader(train_ids))
    run_and_dump(make_loader(val_ids))

    imagesets = os.path.join(out_root, "ImageSets")
    os.makedirs(imagesets, exist_ok=True)
    with open(os.path.join(imagesets, "train.txt"), "w") as f:
        for k in train_ids:
            f.write(k + "\n")
    with open(os.path.join(imagesets, "val.txt"), "w") as f:
        for k in val_ids:
            f.write(k + "\n")

    print("Export complete:", os.path.abspath(velodyne_dir))
    print("Train/val split saved to:", os.path.abspath(imagesets))
    return out_root


def parse_args():
    p = argparse.ArgumentParser(
        description="Export Enhance1024 bins for PointPillars using C4RFNet"
    )
    p.add_argument("--ckpt", default="checkpoints_c4rfnet/best_model.ckpt", help="Path to C4RFNet checkpoint")
    p.add_argument("--base", default="VoD", help="Base directory of VoD dataset")
    p.add_argument("--out", default="output", help="Output directory for enhanced point clouds")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--nw", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_enhance1024(
        model_ckpt=args.ckpt,
        base_dir=args.base,
        out_root=args.out,
        batch_size=args.bs,
        num_workers=args.nw,
        split_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )

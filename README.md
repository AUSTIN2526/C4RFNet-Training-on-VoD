# 🚘 C4RFNet: Radar-LiDAR Fusion on VoD Dataset

C4RFNet is a deep learning model designed to fuse radar and LiDAR data for enhanced 3D perception. It is trained and evaluated on the **VoD (Vehicle on Demand)** dataset using PyTorch, and can export enhanced point clouds for downstream applications such as PointPillars.

---

## 📦 Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.10
* CUDA-compatible GPU (optional but recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
.
├─checkpoints_c4rfnet/         # Model checkpoints
├─data/                        # Data preprocessing modules
├─models/                      # Model definitions (including C4RFNet)
├─output/                      # Exported results
├─VoD/                         # Raw VoD dataset
```

---

## 🏁 Export Enhanced Point Clouds (Enhance1024)

Use a trained C4RFNet model to transform sparse radar input into dense 1024-point clouds:

```bash
python export_enhance1024.py \
  --ckpt checkpoints_c4rfnet/best_model.ckpt \
  --base VoD \
  --out output \
  --bs 16 \
  --nw 4 \
  --val_ratio 0.1 \
  --seed 42 \
  --device cuda
```

**Output files:**

* `output/velodyne/*.bin`: Generated point cloud binary files
* `output/ImageSets/train.txt`, `val.txt`: File ID lists for training and validation splits

---

## 🧠 Model Overview: C4RFNet

| Parameter          | Description              | Default |
| ------------------ | ------------------------ | ------- |
| `enc_dim`          | Encoder output dimension | 1024    |
| `grid_h`, `grid_w` | Feature map resolution   | 32 x 32 |
| `local_k`          | KNN neighborhood size    | 8       |
| `loss_w_global`    | Global loss weight       | 1.0     |
| `loss_w_local`     | Local loss weight        | 1.0     |

---

## 🧪 Training the Model

Basic training command:

```bash
python train_c4rfnet.py
```

With custom parameters:

```bash
python train_c4rfnet.py \
  --base_dir VoD \
  --batch_size 16 \
  --epochs 40 \
  --lr 0.001 \
  --drop_last_train \
  --pin_memory \
  --save_dir ./checkpoints_c4rfnet
```

---

## ⚙️ Configurable Parameters

### Data Settings

| Argument         | Description                      | Default |
| ---------------- | -------------------------------- | ------- |
| `--base_dir`     | Path to VoD dataset              | `VoD`   |
| `--val_ratio`    | Validation split ratio           | `0.1`   |
| `--radar_points` | Number of points per radar frame | `512`   |
| `--lidar_points` | Number of points per LiDAR frame | `1024`  |

### Model & Training Settings

| Argument           | Description                          | Default |
| ------------------ | ------------------------------------ | ------- |
| `--enc_dim`        | Encoder output dimension             | `1024`  |
| `--epochs`         | Total number of training epochs      | `40`    |
| `--lr`             | Learning rate                        | `1e-3`  |
| `--grad_clip`      | Max gradient norm for clipping       | `1.0`   |
| `--early_stopping` | Epochs to wait before early stopping | `5`     |

---

## 🐛 Troubleshooting

* **RuntimeError: data not found**
  Ensure the `VoD/` directory contains valid `lidar/` and `radar/` subfolders.

* **CUDA out of memory**
  Reduce `--batch_size` or `--enc_dim` to lower GPU memory usage.

* **Empty `.bin` output files**
  Check that the checkpoint path is correct and the input data is valid.
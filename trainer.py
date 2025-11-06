# -*- coding: utf-8 -*-
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer, scheduler,
                 train_loader, valid_loader,
                 device=None, epochs=20, grad_clip=None,
                 early_stopping=10, save_dir="./checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.early_stopping = early_stopping

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_path = os.path.join(self.save_dir, "best_model.ckpt")

    def _step_batch(self, batch, train=True):
        radar = batch["radar"].to(self.device)
        image = batch["image"].to(self.device)
        lidar = batch["lidar"].to(self.device)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        recon, _ = self.model(radar, image)
        loss = self.model.loss(recon, lidar)

        if train:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss.detach()

    def train(self, show_loss=True):
        best = float('inf')
        no_improve = 0
        history = {"train": [], "valid": []}

        for epoch in range(1, self.epochs + 1):
            # train
            self.model.train()
            tr_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
            for batch in pbar:
                loss = self._step_batch(batch, train=True)
                tr_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            tr_loss /= max(1, len(self.train_loader))

            # valid
            self.model.eval()
            va_loss = 0.0
            with torch.no_grad():
                pbar = tqdm(self.valid_loader, desc=f"Valid {epoch}", leave=False)
                for batch in pbar:
                    loss = self._step_batch(batch, train=False)
                    va_loss += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            va_loss /= max(1, len(self.valid_loader))

            history["train"].append(tr_loss)
            history["valid"].append(va_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch {epoch:03d} | Train {tr_loss:.5f} | Valid {va_loss:.5f} | LR {self.optimizer.param_groups[0]['lr']:.6f}")

            if va_loss < best:
                best = va_loss
                no_improve = 0
                torch.save(self.model.state_dict(), self.best_path)
                print(f"  -> Save best to {self.best_path} (loss={best:.6f})")
            else:
                no_improve += 1
                if no_improve >= self.early_stopping:
                    print("  -> Early stopping")
                    break

        if show_loss:
            plt.figure()
            plt.plot(history["train"], label="train")
            plt.plot(history["valid"], label="valid")
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        return history

"""
PyTorch Lightning (Flash-Style) Tabular Neural Network for Bank Nifty return prediction.
Implements modular LightningModule tasks for tabular regression and classification.
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torchmetrics


class TabularDataset(Dataset):
    """PyTorch Dataset for Tabular Data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FlashTabularRegressorModule(pl.LightningModule):
    """Lightning Tabular Regressor Task (Flash-style)."""
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.mae_metric = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        mae = self.mae_metric(y_hat, y)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_mae", mae, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class FlashTabularClassifierModule(pl.LightningModule):
    """Lightning Tabular Classifier Task (Flash-style)."""
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_metric = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_clf_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        acc = self.acc_metric(preds, y)
        self.log("val_clf_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_clf_loss"}}


class BankNiftyLightningFlash:
    """PyTorch Lightning Flash-Style Tabular Pipeline."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        pl.seed_everything(random_state, workers=True)
        self.scaler = StandardScaler()
        self.reg_module = None
        self.clf_module = None
        self.training_time = 0.0

    def fit(self, X_train: pd.DataFrame, y_train_reg: pd.Series, y_train_clf: pd.Series,
            X_val: pd.DataFrame = None, y_val_reg: pd.Series = None, y_val_clf: pd.Series = None):
        """Fit Lightning Regressor and Classifier models."""
        start_time = time.time()
        
        X_tr = self.scaler.fit_transform(X_train)
        
        # Prepare validation
        if X_val is not None:
            X_v = self.scaler.transform(X_val)
        else:
            val_split = int(len(X_tr) * 0.85)
            X_v = X_tr[val_split:]
            X_tr = X_tr[:val_split]
            y_val_reg = y_train_reg.iloc[val_split:]
            y_train_reg = y_train_reg.iloc[:val_split]
            y_val_clf = y_train_clf.iloc[val_split:]
            y_train_clf = y_train_clf.iloc[:val_split]

        train_ds_reg = TabularDataset(X_tr, y_train_reg.values)
        val_ds_reg = TabularDataset(X_v, y_val_reg.values)
        train_loader_reg = DataLoader(train_ds_reg, batch_size=32, shuffle=True)
        val_loader_reg = DataLoader(val_ds_reg, batch_size=32, shuffle=False)

        train_ds_clf = TabularDataset(X_tr, y_train_clf.values)
        val_ds_clf = TabularDataset(X_v, y_val_clf.values)
        train_loader_clf = DataLoader(train_ds_clf, batch_size=32, shuffle=True)
        val_loader_clf = DataLoader(val_ds_clf, batch_size=32, shuffle=False)

        input_dim = X_tr.shape[1]

        # 1. Train Regressor
        self.reg_module = FlashTabularRegressorModule(input_dim=input_dim)
        trainer_reg = pl.Trainer(
            max_epochs=45,
            callbacks=[EarlyStopping(monitor="val_loss", patience=12, mode="min")],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False
        )
        trainer_reg.fit(self.reg_module, train_loader_reg, val_loader_reg)

        # 2. Train Classifier
        self.clf_module = FlashTabularClassifierModule(input_dim=input_dim)
        trainer_clf = pl.Trainer(
            max_epochs=45,
            callbacks=[EarlyStopping(monitor="val_clf_loss", patience=12, mode="min")],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False
        )
        trainer_clf.fit(self.clf_module, train_loader_clf, val_loader_clf)

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict using trained Lightning models."""
        X_scaled = self.scaler.transform(X)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.reg_module.eval()
        self.clf_module.eval()
        with torch.no_grad():
            pred_reg = self.reg_module(x_tensor).numpy().flatten()
            logits = self.clf_module(x_tensor)
            pred_proba = torch.sigmoid(logits).numpy().flatten()
            pred_clf = (pred_proba >= 0.5).astype(int)

        return pred_reg, pred_clf, pred_proba

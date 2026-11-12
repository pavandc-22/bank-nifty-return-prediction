"""
PyTorch TabNet implementation for Bank Nifty return prediction.
Utilizes self-attention sequential masks for tabular deep learning.
"""

import time
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier


class BankNiftyTabNet:
    """TabNet Regressor & Classifier with Attention Masks."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.regressor = None
        self.classifier = None
        self.training_time = 0.0
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train_reg: pd.Series, y_train_clf: pd.Series,
            X_val: pd.DataFrame = None, y_val_reg: pd.Series = None, y_val_clf: pd.Series = None):
        """Fit TabNet models with early stopping and sparse attention."""
        self.feature_names = list(X_train.columns)
        start_time = time.time()

        X_tr = self.scaler.fit_transform(X_train).astype(np.float32)
        y_tr_reg = y_train_reg.values.reshape(-1, 1).astype(np.float32)
        y_tr_clf = y_train_clf.values.astype(np.int64)

        eval_set_reg = [(X_tr, y_tr_reg)]
        eval_set_clf = [(X_tr, y_tr_clf)]

        if X_val is not None:
            X_v = self.scaler.transform(X_val).astype(np.float32)
            if y_val_reg is not None:
                eval_set_reg.append((X_v, y_val_reg.values.reshape(-1, 1).astype(np.float32)))
            if y_val_clf is not None:
                eval_set_clf.append((X_v, y_val_clf.values.astype(np.int64)))

        # 1. TabNet Regressor
        self.regressor = TabNetRegressor(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.3,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.8},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            seed=self.random_state,
            verbose=0
        )
        self.regressor.fit(
            X_train=X_tr,
            y_train=y_tr_reg,
            eval_set=eval_set_reg,
            eval_metric=["mae", "mse"],
            max_epochs=60,
            patience=15,
            batch_size=64,
            virtual_batch_size=32
        )

        # 2. TabNet Classifier
        self.classifier = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.3,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.8},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            seed=self.random_state,
            verbose=0
        )
        self.classifier.fit(
            X_train=X_tr,
            y_train=y_tr_clf,
            eval_set=eval_set_clf,
            eval_metric=["auc", "accuracy"],
            max_epochs=60,
            patience=15,
            batch_size=64,
            virtual_batch_size=32
        )

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict test instances using TabNet."""
        X_scaled = self.scaler.transform(X).astype(np.float32)
        pred_reg = self.regressor.predict(X_scaled).flatten()
        pred_proba = self.classifier.predict_proba(X_scaled)[:, 1]
        pred_clf = (pred_proba >= 0.5).astype(int)
        return pred_reg, pred_clf, pred_proba

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Extract global feature importance derived from attention masks."""
        importances = self.regressor.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)

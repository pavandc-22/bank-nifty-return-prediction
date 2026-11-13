"""
Fastai + PyTorch tabular implementation for Bank Nifty return prediction.
"""

import time
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from fastai.tabular.all import (
    TabularPandas, Categorify, Normalize,
    tabular_learner, mse, accuracy, RocAucBinary,
    IndexSplitter, EarlyStoppingCallback, SaveModelCallback
)


class BankNiftyFastAI:
    """FastAI Tabular Learner with PyTorch embeddings and 1Cycle policy."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.reg_learner = None
        self.clf_learner = None
        self.training_time = 0.0
        self.cat_names = ["day_of_week", "month", "is_month_end", "is_quarter_end"]
        self.cont_names = []

    def fit(self, train_df: pd.DataFrame, feature_cols: list, val_df: pd.DataFrame = None):
        """Fit fastai tabular regression and classification learners."""
        start_time = time.time()
        torch.manual_seed(self.random_state)

        # Separate categorical and continuous columns
        self.cat_names = [c for c in self.cat_names if c in feature_cols]
        self.cont_names = [c for c in feature_cols if c not in self.cat_names]

        # Combine train and val for fastai IndexSplitter
        if val_df is not None:
            full_df = pd.concat([train_df, val_df], ignore_index=True)
            splits = IndexSplitter(list(range(len(train_df), len(full_df))))(full_df)
        else:
            full_df = train_df.copy()
            val_idx = list(range(int(len(full_df) * 0.8), len(full_df)))
            splits = IndexSplitter(val_idx)(full_df)

        procs = [Categorify, Normalize]

        # 1. Regression Learner
        to_reg = TabularPandas(
            full_df,
            procs=procs,
            cat_names=self.cat_names,
            cont_names=self.cont_names,
            y_names="target_return_next_1d",
            y_block=None,
            splits=splits
        )
        dls_reg = to_reg.dataloaders(bs=32)
        self.reg_learner = tabular_learner(
            dls_reg,
            layers=[128, 64, 32],
            loss_func=mse,
            metrics=None
        )
        self.reg_learner.fit_one_cycle(15, lr_max=1e-3)

        # 2. Classification Learner (Direction)
        full_df_clf = full_df.copy()
        full_df_clf["target_direction_next_1d"] = full_df_clf["target_direction_next_1d"].astype(int)
        
        to_clf = TabularPandas(
            full_df_clf,
            procs=procs,
            cat_names=self.cat_names,
            cont_names=self.cont_names,
            y_names="target_direction_next_1d",
            splits=splits
        )
        dls_clf = to_clf.dataloaders(bs=32)
        self.clf_learner = tabular_learner(
            dls_clf,
            layers=[128, 64, 32],
            metrics=[accuracy]
        )
        self.clf_learner.fit_one_cycle(15, lr_max=1e-3)

        self.training_time = time.time() - start_time
        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict test targets using fastai learners."""
        # Regression predictions
        dl_reg = self.reg_learner.dls.test_dl(test_df)
        preds_reg, _ = self.reg_learner.get_preds(dl=dl_reg)
        pred_reg = preds_reg.numpy().flatten()

        # Classification predictions
        dl_clf = self.clf_learner.dls.test_dl(test_df)
        preds_clf, _ = self.clf_learner.get_preds(dl=dl_clf)
        # fastai returns softmax probabilities [p0, p1]
        probs = preds_clf.numpy()
        pred_proba = probs[:, 1] if probs.shape[1] > 1 else probs.flatten()
        pred_clf = (pred_proba >= 0.5).astype(int)

        return pred_reg, pred_clf, pred_proba

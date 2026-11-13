"""
XGBoost implementation for Bank Nifty return regression and classification.
"""

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple


class BankNiftyXGBoost:
    """XGBoost Model for Bank Nifty Return Prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.regressor = None
        self.classifier = None
        self.training_time = 0.0
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train_reg: pd.Series, y_train_clf: pd.Series,
            X_val: pd.DataFrame = None, y_val_reg: pd.Series = None, y_val_clf: pd.Series = None):
        """Fit both regression and classification models."""
        self.feature_names = list(X_train.columns)
        start_time = time.time()

        # 1. Regression Model
        self.regressor = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1
        )
        
        eval_set = [(X_train, y_train_reg)]
        if X_val is not None and y_val_reg is not None:
            eval_set.append((X_val, y_val_reg))
            
        self.regressor.fit(
            X_train, y_train_reg,
            eval_set=eval_set,
            verbose=False
        )

        # 2. Classification Model (Direction: Up/Down)
        self.classifier = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1
        )
        
        clf_eval_set = [(X_train, y_train_clf)]
        if X_val is not None and y_val_clf is not None:
            clf_eval_set.append((X_val, y_val_clf))
            
        self.classifier.fit(
            X_train, y_train_clf,
            eval_set=clf_eval_set,
            verbose=False
        )

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict returns and direction."""
        pred_reg = self.regressor.predict(X)
        pred_clf = self.classifier.predict(X)
        pred_proba = self.classifier.predict_proba(X)[:, 1]
        return pred_reg, pred_clf, pred_proba

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Extract top feature importances by Gain."""
        importances = self.regressor.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)

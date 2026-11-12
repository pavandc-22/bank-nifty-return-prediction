"""
Keras 3 / TensorFlow Deep Learning implementation for Bank Nifty return prediction.
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


class BankNiftyKeras:
    """Deep Neural Network in Keras for Bank Nifty Return Prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.reg_model = None
        self.clf_model = None
        self.training_time = 0.0

    def _build_regressor(self, input_dim: int) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="swish", kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(64, activation="swish", kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="swish"),
            layers.Dense(1, activation="linear")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.Huber(delta=1.0),
            metrics=["mae", "mse"]
        )
        return model

    def _build_classifier(self, input_dim: int) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="swish", kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(64, activation="swish", kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="swish"),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")]
        )
        return model

    def fit(self, X_train: pd.DataFrame, y_train_reg: pd.Series, y_train_clf: pd.Series,
            X_val: pd.DataFrame = None, y_val_reg: pd.Series = None, y_val_clf: pd.Series = None):
        """Train Keras regression and classification neural nets."""
        start_time = time.time()
        
        # Scale features
        X_tr = self.scaler.fit_transform(X_train)
        val_data_reg = None
        val_data_clf = None
        if X_val is not None:
            X_v = self.scaler.transform(X_val)
            if y_val_reg is not None:
                val_data_reg = (X_v, y_val_reg.values)
            if y_val_clf is not None:
                val_data_clf = (X_v, y_val_clf.values)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_data_reg else "loss",
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_data_reg else "loss",
            factor=0.5,
            patience=7,
            min_lr=1e-5
        )

        input_dim = X_tr.shape[1]
        
        # Train Regressor
        self.reg_model = self._build_regressor(input_dim)
        self.reg_model.fit(
            X_tr, y_train_reg.values,
            validation_data=val_data_reg,
            epochs=80,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Train Classifier
        self.clf_model = self._build_classifier(input_dim)
        self.clf_model.fit(
            X_tr, y_train_clf.values,
            validation_data=val_data_clf,
            epochs=80,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions from trained Keras models."""
        X_scaled = self.scaler.transform(X)
        pred_reg = self.reg_model.predict(X_scaled, verbose=0).flatten()
        pred_proba = self.clf_model.predict(X_scaled, verbose=0).flatten()
        pred_clf = (pred_proba >= 0.5).astype(int)
        return pred_reg, pred_clf, pred_proba

"""
Master Execution Pipeline for Bank Nifty Return Prediction.
Trains, evaluates, and benchmarks all 5 frameworks:
1. XGBoost
2. Keras
3. PyTorch with FastAI
4. PyTorch with TabNet
5. PyTorch with Lightning (Flash-Style)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".matplotlib_tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from src.feature_engineering import prepare_datasets
from src.eda import perform_full_eda
from src.models.xgboost_model import BankNiftyXGBoost
from src.models.keras_model import BankNiftyKeras
from src.models.fastai_model import BankNiftyFastAI
from src.models.tabnet_model import BankNiftyTabNet
from src.models.lightning_model import BankNiftyLightningFlash
from src.evaluation import (
    calculate_metrics, plot_comparison_metrics,
    plot_predicted_vs_actual, plot_top_features
)


def run_full_benchmark():
    print("=================================================================")
    print(" BANK NIFTY PERFORMANCE RETURN PREDICTION - MULTI-FRAMEWORK PIPELINE ")
    print("=================================================================\n")

    # 1. Check data availability and prepare datasets
    processed_train = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    processed_test = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

    if not os.path.exists(processed_train) or not os.path.exists(processed_test):
        print("Processed datasets not found. Running feature engineering...")
        train_df, test_df, feature_cols = prepare_datasets(test_ratio=0.2)
    else:
        print("Loading pre-processed datasets...")
        train_df = pd.read_csv(processed_train)
        test_df = pd.read_csv(processed_test)
        train_df["price_date"] = pd.to_datetime(train_df["price_date"])
        test_df["price_date"] = pd.to_datetime(test_df["price_date"])
        
        non_features = [
            "symbol", "stock", "ticker", "price_date",
            "open", "high", "low", "close", "volume", "dividends",
            "target_return_next_1d", "target_direction_next_1d"
        ]
        feature_cols = [c for c in train_df.columns if c not in non_features]

    print(f"Dataset summary: Train={len(train_df)} rows, Test={len(test_df)} rows, Features={len(feature_cols)}")

    # Extract feature matrices and target vectors
    X_train = train_df[feature_cols]
    y_train_reg = train_df["target_return_next_1d"]
    y_train_clf = train_df["target_direction_next_1d"]

    X_test = test_df[feature_cols]
    y_test_reg = test_df["target_return_next_1d"].values
    y_test_clf = test_df["target_direction_next_1d"].values
    test_dates = test_df["price_date"]

    # In-sample validation split for neural networks & early stopping (last 15% of train set)
    val_split_idx = int(len(train_df) * 0.85)
    X_tr = X_train.iloc[:val_split_idx]
    y_tr_reg = y_train_reg.iloc[:val_split_idx]
    y_tr_clf = y_train_clf.iloc[:val_split_idx]

    X_val = X_train.iloc[val_split_idx:]
    y_val_reg = y_train_reg.iloc[val_split_idx:]
    y_val_clf = y_train_clf.iloc[val_split_idx:]

    results = []
    predictions_reg = {}
    top_feature_df = None

    # -------------------------------------------------------------
    # Model 1: XGBoost
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print(">> Training Model 1: XGBoost...")
    print("-------------------------------------------------------------")
    xgb_model = BankNiftyXGBoost(random_state=42)
    xgb_model.fit(X_tr, y_tr_reg, y_tr_clf, X_val, y_val_reg, y_val_clf)
    p_reg, p_clf, p_prob = xgb_model.predict(X_test)
    predictions_reg["XGBoost"] = p_reg
    top_feature_df = xgb_model.get_feature_importance(top_n=15)
    
    m_xgb = calculate_metrics(y_test_reg, p_reg, y_test_clf, p_clf, p_prob, xgb_model.training_time)
    m_xgb["Model"] = "XGBoost"
    m_xgb["Simplicity"] = "High (Clean scikit-learn API, minimal boilerplate)"
    m_xgb["Transparency"] = "High (Native feature importances, tree split gain/weight)"
    m_xgb["Efficacy"] = "High (Sub-second training, optimal gradient boosting)"
    results.append(m_xgb)
    print(f"XGBoost Test Metrics: RMSE={m_xgb['RMSE']:.4f}, MAE={m_xgb['MAE']:.4f}, R2={m_xgb['R2']:.4f}, Dir_Acc={m_xgb['Directional_Accuracy_Pct']:.2f}%, TrainTime={m_xgb['Training_Time_Sec']:.2f}s")

    # -------------------------------------------------------------
    # Model 2: Keras Deep Neural Network
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print(">> Training Model 2: Keras (TensorFlow Backend)...")
    print("-------------------------------------------------------------")
    keras_model = BankNiftyKeras(random_state=42)
    keras_model.fit(X_tr, y_tr_reg, y_tr_clf, X_val, y_val_reg, y_val_clf)
    p_reg, p_clf, p_prob = keras_model.predict(X_test)
    predictions_reg["Keras"] = p_reg

    m_keras = calculate_metrics(y_test_reg, p_reg, y_test_clf, p_clf, p_prob, keras_model.training_time)
    m_keras["Model"] = "Keras"
    m_keras["Simplicity"] = "Moderate (Requires scaling, layer wiring, callbacks)"
    m_keras["Transparency"] = "Low (Deep black-box neural representations)"
    m_keras["Efficacy"] = "Moderate (High expressiveness, moderate GPU/CPU compute)"
    results.append(m_keras)
    print(f"Keras Test Metrics: RMSE={m_keras['RMSE']:.4f}, MAE={m_keras['MAE']:.4f}, R2={m_keras['R2']:.4f}, Dir_Acc={m_keras['Directional_Accuracy_Pct']:.2f}%, TrainTime={m_keras['Training_Time_Sec']:.2f}s")

    # -------------------------------------------------------------
    # Model 3: PyTorch with FastAI
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print(">> Training Model 3: PyTorch with FastAI Tabular Learner...")
    print("-------------------------------------------------------------")
    fastai_model = BankNiftyFastAI(random_state=42)
    # fastai uses the train_df slice and feature cols
    fastai_model.fit(train_df.iloc[:val_split_idx], feature_cols, train_df.iloc[val_split_idx:])
    p_reg, p_clf, p_prob = fastai_model.predict(test_df)
    predictions_reg["FastAI"] = p_reg

    m_fastai = calculate_metrics(y_test_reg, p_reg, y_test_clf, p_clf, p_prob, fastai_model.training_time)
    m_fastai["Model"] = "FastAI"
    m_fastai["Simplicity"] = "Moderate-High (1Cycle policy, automated embedding layers)"
    m_fastai["Transparency"] = "Low-Moderate (Neural embeddings, difficult feature attribution)"
    m_fastai["Efficacy"] = "Moderate (Fast 1Cycle convergence, PyTorch backend)"
    results.append(m_fastai)
    print(f"FastAI Test Metrics: RMSE={m_fastai['RMSE']:.4f}, MAE={m_fastai['MAE']:.4f}, R2={m_fastai['R2']:.4f}, Dir_Acc={m_fastai['Directional_Accuracy_Pct']:.2f}%, TrainTime={m_fastai['Training_Time_Sec']:.2f}s")

    # -------------------------------------------------------------
    # Model 4: PyTorch with TabNet
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print(">> Training Model 4: PyTorch with TabNet...")
    print("-------------------------------------------------------------")
    tabnet_model = BankNiftyTabNet(random_state=42)
    tabnet_model.fit(X_tr, y_tr_reg, y_tr_clf, X_val, y_val_reg, y_val_clf)
    p_reg, p_clf, p_prob = tabnet_model.predict(X_test)
    predictions_reg["TabNet"] = p_reg

    m_tabnet = calculate_metrics(y_test_reg, p_reg, y_test_clf, p_clf, p_prob, tabnet_model.training_time)
    m_tabnet["Model"] = "TabNet"
    m_tabnet["Simplicity"] = "Moderate (Custom TabNet hyperparams: n_d, n_a, gamma)"
    m_tabnet["Transparency"] = "High (Sparse attention masks reveal feature selection)"
    m_tabnet["Efficacy"] = "Moderate (Iterative sequential attention steps, compute-intensive)"
    results.append(m_tabnet)
    print(f"TabNet Test Metrics: RMSE={m_tabnet['RMSE']:.4f}, MAE={m_tabnet['MAE']:.4f}, R2={m_tabnet['R2']:.4f}, Dir_Acc={m_tabnet['Directional_Accuracy_Pct']:.2f}%, TrainTime={m_tabnet['Training_Time_Sec']:.2f}s")

    # -------------------------------------------------------------
    # Model 5: PyTorch Lightning (Flash Tabular Style)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print(">> Training Model 5: PyTorch Lightning (Flash Style)...")
    print("-------------------------------------------------------------")
    pl_model = BankNiftyLightningFlash(random_state=42)
    pl_model.fit(X_tr, y_tr_reg, y_tr_clf, X_val, y_val_reg, y_val_clf)
    p_reg, p_clf, p_prob = pl_model.predict(X_test)
    predictions_reg["PyTorch Lightning"] = p_reg

    m_pl = calculate_metrics(y_test_reg, p_reg, y_test_clf, p_clf, p_prob, pl_model.training_time)
    m_pl["Model"] = "PyTorch Lightning"
    m_pl["Simplicity"] = "Complex (Structured LightningModule, Trainer loops, Callbacks)"
    m_pl["Transparency"] = "Low (Deep MLP/Torchmetrics, black-box hidden state)"
    m_pl["Efficacy"] = "High (Industrial engineering, auto GPU/CPU scaling, checkpointing)"
    results.append(m_pl)
    print(f"Lightning Test Metrics: RMSE={m_pl['RMSE']:.4f}, MAE={m_pl['MAE']:.4f}, R2={m_pl['R2']:.4f}, Dir_Acc={m_pl['Directional_Accuracy_Pct']:.2f}%, TrainTime={m_pl['Training_Time_Sec']:.2f}s")

    # -------------------------------------------------------------
    # Summarize and Export Results
    # -------------------------------------------------------------
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    display_cols = [
        "Model", "RMSE", "MAE", "R2", "Directional_Accuracy_Pct",
        "Pearson_Corr", "Classification_Accuracy_Pct", "ROC_AUC",
        "Training_Time_Sec", "Simplicity", "Transparency", "Efficacy"
    ]
    results_df = results_df[display_cols]
    
    print("\n=============================================================")
    print(" FINAL MULTI-FRAMEWORK BENCHMARK SUMMARY ")
    print("=============================================================")
    print(results_df.to_string(index=False))

    # Save summary CSV
    csv_path = os.path.join(PROJECT_ROOT, "outputs", "metrics_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics summary table to: {csv_path}")

    # Qualitative comparison table
    qual_df = results_df[["Model", "Simplicity", "Transparency", "Efficacy", "Training_Time_Sec", "Directional_Accuracy_Pct"]]
    qual_path = os.path.join(PROJECT_ROOT, "outputs", "qualitative_comparison.csv")
    qual_df.to_csv(qual_path, index=False)

    # -------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------
    print("\nGenerating final comparison plots...")
    plot_comparison_metrics(results_df)
    plot_predicted_vs_actual(test_dates, y_test_reg, predictions_reg)
    if top_feature_df is not None:
        plot_top_features(top_feature_df, model_name="XGBoost")

    print("\nAll model training, evaluations, and visualizations successfully completed!")
    return results_df


if __name__ == "__main__":
    run_full_benchmark()

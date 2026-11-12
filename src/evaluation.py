"""
Model evaluation, benchmark comparison, and metric reporting module.
Evaluates the 5 models on quantitative metrics (RMSE, R2, MAE, Hit Ratio, AUC)
and qualitative dimensions (Simplicity, Transparency, Efficacy).
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import pearsonr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def calculate_metrics(y_true_reg: np.ndarray, y_pred_reg: np.ndarray,
                      y_true_clf: np.ndarray, y_pred_clf: np.ndarray,
                      y_pred_proba: np.ndarray, training_time: float) -> dict:
    """Compute comprehensive regression, classification, and efficiency metrics."""
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    
    # Financial directional accuracy (sign hit ratio)
    sign_true = np.sign(y_true_reg)
    sign_pred = np.sign(y_pred_reg)
    dir_acc = np.mean(sign_true == sign_pred) * 100.0
    
    # Pearson correlation
    corr, _ = pearsonr(y_true_reg, y_pred_reg)

    # Classification metrics
    acc = accuracy_score(y_true_clf, y_pred_clf) * 100.0
    prec = precision_score(y_true_clf, y_pred_clf, zero_division=0) * 100.0
    rec = recall_score(y_true_clf, y_pred_clf, zero_division=0) * 100.0
    f1 = f1_score(y_true_clf, y_pred_clf, zero_division=0) * 100.0
    try:
        auc = roc_auc_score(y_true_clf, y_pred_proba)
    except Exception:
        auc = 0.5

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Directional_Accuracy_Pct": float(dir_acc),
        "Pearson_Corr": float(corr),
        "Classification_Accuracy_Pct": float(acc),
        "Precision_Pct": float(prec),
        "Recall_Pct": float(rec),
        "F1_Score_Pct": float(f1),
        "ROC_AUC": float(auc),
        "Training_Time_Sec": float(training_time)
    }


def plot_comparison_metrics(results_df: pd.DataFrame):
    """Generate bar chart comparing all 5 models across core quantitative metrics."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = results_df["Model"].tolist()
    palette = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]

    # 1. RMSE (Lower is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, results_df["RMSE"], color=palette, alpha=0.85, width=0.55)
    ax1.set_title("Root Mean Squared Error (RMSE) - Lower is Better", fontsize=12, fontweight="bold")
    ax1.set_ylabel("RMSE (%)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2., b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # 2. MAE (Lower is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, results_df["MAE"], color=palette, alpha=0.85, width=0.55)
    ax2.set_title("Mean Absolute Error (MAE) - Lower is Better", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MAE (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # 3. R2 Score (Higher is better)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, results_df["R2"], color=palette, alpha=0.85, width=0.55)
    ax3.set_title("Coefficient of Determination (R²) - Higher is Better", fontsize=12, fontweight="bold")
    ax3.set_ylabel("R² Score", fontsize=10)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.grid(True, linestyle=":", alpha=0.6)
    for b in bars3:
        val = b.get_height()
        va_pos = "bottom" if val >= 0 else "top"
        ax3.text(b.get_x() + b.get_width()/2., val + (0.005 if val >= 0 else -0.01), f"{val:.3f}", ha="center", va=va_pos, fontsize=9)

    # 4. Directional Accuracy (Higher is better)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, results_df["Directional_Accuracy_Pct"], color=palette, alpha=0.85, width=0.55)
    ax4.axhline(50.0, color="gray", linestyle="--", linewidth=1.2, label="Random Baseline (50%)")
    ax4.set_title("Directional Return Hit Ratio (%) - Higher is Better", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Directional Accuracy (%)", fontsize=10)
    ax4.set_ylim(40, max(65, results_df["Directional_Accuracy_Pct"].max() + 5))
    ax4.legend(loc="lower right")
    ax4.grid(True, linestyle=":", alpha=0.6)
    for b in bars4:
        ax4.text(b.get_x() + b.get_width()/2., b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    for ax in axes.flat:
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)

    plt.tight_layout()
    comp_path = os.path.join(FIGURES_DIR, "06_model_comparison_metrics.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")


def plot_predicted_vs_actual(dates: pd.Series, y_true: np.ndarray, predictions: dict):
    """Plot out-of-sample time-series trajectory of actual vs predicted next-day returns."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(15, 7))

    plt.plot(dates, y_true, label="Actual Bank Nifty Return (%)", color="black", linewidth=1.8, alpha=0.8)
    colors = {"XGBoost": "#2b5c8f", "Keras": "#e67e22", "FastAI": "#27ae60", "TabNet": "#8e44ad", "PyTorch Lightning": "#c0392b"}

    for name, preds in predictions.items():
        plt.plot(dates, preds, label=f"{name} Pred", linestyle="--", alpha=0.75, linewidth=1.2, color=colors.get(name, None))

    plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.title("Out-of-Sample Test Set: Actual vs. Predicted Next-Day Returns across All 5 Models", fontsize=13, fontweight="bold")
    plt.xlabel("Trading Date", fontsize=11)
    plt.ylabel("Return (%)", fontsize=11)
    plt.legend(loc="upper right", frameon=True, ncol=3)
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    pred_path = os.path.join(FIGURES_DIR, "07_predicted_vs_actual.png")
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"Saved: {pred_path}")


def plot_top_features(feature_importance_df: pd.DataFrame, model_name: str = "XGBoost"):
    """Plot horizontal bar chart of top predictive features."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=feature_importance_df,
        y="feature",
        x="importance",
        palette="Blues_r"
    )
    plt.title(f"Top 15 Most Influential Features for Bank Nifty Return Prediction ({model_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance (Gain / Weight)", fontsize=10)
    plt.ylabel("Feature Indicator", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    feat_path = os.path.join(FIGURES_DIR, "08_feature_importance.png")
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"Saved: {feat_path}")

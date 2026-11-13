"""
Exploratory Data Analysis (EDA) for Bank Nifty Return Prediction.
Performs statistical profiling, econometric testing (ADF, KPSS, Jarque-Bera),
volatility clustering analysis, correlation heatmaps, and publication-ready charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def run_statistical_tests(series: pd.Series, name: str = "Series") -> dict:
    """Run ADF, KPSS, and Jarque-Bera statistical tests."""
    clean = series.dropna()
    
    # 1. ADF Test
    adf_res = adfuller(clean, autolag="AIC")
    
    # 2. KPSS Test
    try:
        kpss_res = kpss(clean, regression="c", nlags="auto")
    except Exception:
        kpss_res = (np.nan, np.nan, {}, {})
        
    # 3. Jarque-Bera Test
    jb_stat, jb_p = stats.jarque_bera(clean)
    
    # Descriptive metrics
    mean_val = clean.mean()
    std_val = clean.std()
    skew_val = clean.skew()
    kurt_val = clean.kurtosis() # Excess kurtosis

    return {
        "variable": name,
        "count": len(clean),
        "mean": mean_val,
        "std": std_val,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "adf_stat": adf_res[0],
        "adf_pvalue": adf_res[1],
        "adf_stationary": adf_res[1] < 0.05,
        "kpss_stat": kpss_res[0],
        "kpss_pvalue": kpss_res[1],
        "jb_stat": jb_stat,
        "jb_pvalue": jb_p,
        "is_normal": jb_p > 0.05
    }


def perform_full_eda():
    """Execute complete EDA pipeline and save charts & statistical summaries."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load processed data
    df_path = os.path.join(PROCESSED_DIR, "bank_nifty_engineered.csv")
    df = pd.read_csv(df_path)
    df["price_date"] = pd.to_datetime(df["price_date"])

    print("=== 1. Summary Statistics & Econometric Tests ===")
    price_stats = run_statistical_tests(df["close"], "Bank Nifty Close Price")
    return_stats = run_statistical_tests(df["ret_1d"], "Bank Nifty 1-Day Return (%)")
    target_stats = run_statistical_tests(df["target_return_next_1d"], "Target Next-Day Return (%)")

    stats_summary = pd.DataFrame([price_stats, return_stats, target_stats])
    print(stats_summary.to_string())

    # Save summary table
    stats_summary_path = os.path.join(PROJECT_ROOT, "outputs", "eda_statistical_summary.csv")
    stats_summary.to_csv(stats_summary_path, index=False)

    # -------------------------------------------------------------
    # Figure 1: Historical Price Trajectory & Moving Averages
    # -------------------------------------------------------------
    print("\nGenerating Figure 1: Price and Moving Average Regime...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    ax1.plot(df["price_date"], df["close"], label="Bank Nifty Close", color="#1f77b4", linewidth=1.8)
    sma_50 = df["close"].rolling(50).mean()
    sma_200 = df["close"].rolling(200).mean()
    ax1.plot(df["price_date"], sma_50, label="50-Day SMA", color="#ff7f0e", linestyle="--", alpha=0.85)
    ax1.plot(df["price_date"], sma_200, label="200-Day SMA", color="#2ca02c", linestyle="-.", alpha=0.85)
    
    ax1.set_title("Nifty Bank Index (^NSEBANK) - 7-Year Historical Price Dynamics & Trend Regimes", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Index Level (INR)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Highlight market regimes
    ax1.axvspan(pd.to_datetime("2020-03-01"), pd.to_datetime("2020-06-30"), color="red", alpha=0.1, label="COVID Shock")
    ax1.axvspan(pd.to_datetime("2020-07-01"), pd.to_datetime("2021-12-31"), color="green", alpha=0.08, label="Post-COVID Expansion")
    
    # Subplot 2: Daily percentage return
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in df["ret_1d"]]
    ax2.bar(df["price_date"], df["ret_1d"], color=colors, width=1.5, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Daily Return (%)", fontsize=11)
    ax2.set_xlabel("Trading Date", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "01_price_and_volume.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Return Distribution, Fat Tails & Normality
    # -------------------------------------------------------------
    print("Generating Figure 2: Return Distribution & Fat Tail Profiling...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    returns = df["ret_1d"].dropna()
    mu, std = returns.mean(), returns.std()
    
    # Histogram + KDE + Normal distribution fit
    sns.histplot(returns, kde=True, stat="density", bins=60, color="#2b5c8f", alpha=0.6, ax=ax1, label="Empirical Returns")
    x = np.linspace(returns.min(), returns.max(), 500)
    p = stats.norm.pdf(x, mu, std)
    ax1.plot(x, p, "r--", linewidth=2, label=f"Normal Fit (μ={mu:.2f}%, σ={std:.2f}%)")
    
    # Empirical vs normal text annotation
    textstr = f"Mean: {mu:.2f}%\nStd: {std:.2f}%\nSkew: {returns.skew():.2f}\nKurt: {returns.kurtosis():.2f}\nJB p-val: {return_stats['jb_pvalue']:.2e}\n(Fat-Tailed / Leptokurtic)"
    ax1.text(0.03, 0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#ccc"))
    
    ax1.set_title("Empirical Return Distribution vs. Normal Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Daily Return (%)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Q-Q Plot
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color("#2b5c8f")
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[1].set_color("red")
    ax2.get_lines()[1].set_linewidth(2)
    ax2.set_title("Normal Q-Q Plot (Showing Significant Tail Divergence)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "02_returns_distribution.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Correlation Matrix of Constituents and Index
    # -------------------------------------------------------------
    print("Generating Figure 3: Cross-Sectional Banking Correlation Heatmap...")
    ret_cols = [c for c in df.columns if c.startswith("ret_1d_")]
    corr_df = df[["ret_1d"] + ret_cols].copy()
    rename_dict = {"ret_1d": "BANKNIFTY"}
    for c in ret_cols:
        rename_dict[c] = c.replace("ret_1d_", "")
    corr_df = corr_df.rename(columns=rename_dict)

    corr_matrix = corr_df.corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0.3, vmax=1.0,
                square=True, linewidths=0.6, cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"})
    plt.title("Correlation Matrix: Bank Nifty Index vs Top Banking Constituents", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "03_correlation_heatmap.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Volatility Clustering & ARCH Effect
    # -------------------------------------------------------------
    print("Generating Figure 4: Volatility Clustering & Risk Regimes...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    # Rolling 20-day annualized volatility (assume 252 trading days)
    ann_vol_20 = df["ret_1d"].rolling(20).std() * np.sqrt(252)
    ann_vol_60 = df["ret_1d"].rolling(60).std() * np.sqrt(252)

    ax1.plot(df["price_date"], ann_vol_20, color="#e74c3c", linewidth=1.5, label="20-Day Annualized Volatility (%)")
    ax1.plot(df["price_date"], ann_vol_60, color="#34495e", linewidth=1.5, linestyle="--", label="60-Day Annualized Volatility (%)")
    ax1.set_title("Bank Nifty Annualized Rolling Volatility (Evidence of Volatility Clustering / Heteroskedasticity)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Annualized Vol (%)", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Intraday Parkinson Volatility vs Close-to-Close Volatility
    ax2.plot(df["price_date"], df["parkinson_vol_10d"], color="#8e44ad", linewidth=1.4, label="10-Day Parkinson Volatility (High-Low Range)")
    ax2.plot(df["price_date"], df["volatility_10d"], color="#2980b9", linewidth=1.2, linestyle=":", label="10-Day Realized Close-to-Close Volatility")
    ax2.set_title("Parkinson High-Low Volatility Estimator vs. Realized Volatility", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Volatility (%)", fontsize=11)
    ax2.set_xlabel("Trading Date", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "04_volatility_clustering.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Feature Dynamics (RSI, MACD, PSU vs Pvt Spread)
    # -------------------------------------------------------------
    print("Generating Figure 5: Technical & Sector Indicators...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # 1. RSI
    ax1.plot(df["price_date"], df["rsi_14"], color="#2980b9", linewidth=1.3, label="14-Day RSI")
    ax1.axhline(70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)")
    ax1.axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
    ax1.set_title("Relative Strength Index (RSI 14) Dynamics", fontsize=12, fontweight="bold")
    ax1.set_ylabel("RSI Value", fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. MACD
    ax2.plot(df["price_date"], df["macd_line"], color="#27ae60", linewidth=1.3, label="MACD Line")
    ax2.plot(df["price_date"], df["macd_signal"], color="#e67e22", linewidth=1.3, linestyle="--", label="MACD Signal (9-day)")
    ax2.bar(df["price_date"], df["macd_diff"], color="#7f8c8d", alpha=0.5, width=1.2, label="Histogram")
    ax2.set_title("Normalized MACD Oscillators", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MACD (%)", fontsize=10)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Private vs PSU Bank Return Spread
    if "pvt_minus_psu_spread" in df.columns:
        spread_rolling = df["pvt_minus_psu_spread"].rolling(20).mean()
        ax3.plot(df["price_date"], spread_rolling, color="#8e44ad", linewidth=1.5, label="20-Day Rolling Private vs. PSU Bank Spread")
        ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax3.set_title("Sector Structural Signal: Private Bank vs. PSU Bank Performance Spread (20-Day Trend)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Spread (%)", fontsize=10)
        ax3.set_xlabel("Trading Date", fontsize=10)
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "05_technical_indicators.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    print("\nEDA completed successfully! All figures generated and saved.")


if __name__ == "__main__":
    perform_full_eda()

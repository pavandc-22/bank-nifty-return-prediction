"""
Feature Engineering module for Bank Nifty return prediction.
Calculates technical indicators, volatility estimators, constituent momentum,
breadth metrics, and calendar features with strict time-series alignment.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Normalized Average True Range (NATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    natr = (atr / close) * 100.0
    return natr


def build_constituent_features(constituents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract constituent bank returns, PSU vs Private bank spreads,
    and market breadth indicators.
    """
    pivot_close = constituents_df.pivot(index="price_date", columns="symbol", values="close")
    pivot_returns = pivot_close.pct_change() * 100.0

    features = pd.DataFrame(index=pivot_returns.index)

    # Individual constituent returns
    for sym in pivot_returns.columns:
        features[f"ret_1d_{sym}"] = pivot_returns[sym]
        features[f"ret_5d_{sym}"] = pivot_close[sym].pct_change(5) * 100.0

    # Sector breadth (% of bank stocks positive today)
    features["bank_breadth_positive_ratio"] = (pivot_returns > 0).mean(axis=1)

    # Sector momentum dispersion (std of returns across banks)
    features["bank_return_dispersion"] = pivot_returns.std(axis=1)

    # Private vs PSU Bank Return Spread
    pvt_banks = [c for c in ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "FEDERALBNK"] if c in pivot_returns.columns]
    psu_banks = [c for c in ["SBIN", "BANKBARODA", "PNB"] if c in pivot_returns.columns]

    if pvt_banks and psu_banks:
        features["pvt_banks_mean_ret"] = pivot_returns[pvt_banks].mean(axis=1)
        features["psu_banks_mean_ret"] = pivot_returns[psu_banks].mean(axis=1)
        features["pvt_minus_psu_spread"] = features["pvt_banks_mean_ret"] - features["psu_banks_mean_ret"]

    # Weighted proxy return (top 5 weight approximation)
    weights = {"HDFCBANK": 0.28, "ICICIBANK": 0.24, "SBIN": 0.11, "KOTAKBANK": 0.10, "AXISBANK": 0.10}
    weighted_ret = pd.Series(0.0, index=pivot_returns.index)
    total_w = 0.0
    for sym, w in weights.items():
        if sym in pivot_returns.columns:
            weighted_ret += pivot_returns[sym] * w
            total_w += w
    if total_w > 0:
        features["constituent_weighted_ret"] = weighted_ret / total_w

    return features.reset_index()


def engineer_bank_nifty_features(raw_index_path: str, raw_constituents_path: str) -> pd.DataFrame:
    """
    Generate comprehensive features combining Bank Nifty index dynamics
    and constituent banking stock fundamentals.
    """
    df = pd.read_csv(raw_index_path)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df = df.sort_values("price_date").reset_index(drop=True)

    # 1. Base Returns
    df["ret_1d"] = df["close"].pct_change(1) * 100.0
    df["ret_2d"] = df["close"].pct_change(2) * 100.0
    df["ret_3d"] = df["close"].pct_change(3) * 100.0
    df["ret_5d"] = df["close"].pct_change(5) * 100.0
    df["ret_10d"] = df["close"].pct_change(10) * 100.0
    df["ret_20d"] = df["close"].pct_change(20) * 100.0
    df["log_ret_1d"] = (np.log(df["close"] / df["close"].shift(1))) * 100.0

    # 2. Moving Average Ratios (Trend)
    for w in [5, 10, 20, 50, 200]:
        sma = df["close"].rolling(w).mean()
        df[f"close_to_sma_{w}"] = (df["close"] / sma - 1.0) * 100.0

    # 3. Exponential Moving Averages & MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ((ema_12 - ema_26) / df["close"]) * 100.0
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_diff"] = macd_line - macd_signal

    # 4. Volatility & Risk Metrics
    df["volatility_5d"] = df["ret_1d"].rolling(5).std()
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()
    df["volatility_60d"] = df["ret_1d"].rolling(60).std()

    # Parkinson Volatility (High-Low range based)
    hl_ratio = np.log(df["high"] / df["low"])
    df["parkinson_vol_10d"] = np.sqrt((hl_ratio ** 2).rolling(10).mean() / (4 * np.log(2))) * 100.0

    # Intraday Candle Geometry
    candle_range = (df["high"] - df["low"]).clip(lower=1e-5)
    df["intraday_range_pct"] = (candle_range / df["close"]) * 100.0
    df["candle_body_ratio"] = (df["close"] - df["open"]) / candle_range
    df["upper_shadow_ratio"] = (df["high"] - np.maximum(df["close"], df["open"])) / candle_range
    df["lower_shadow_ratio"] = (np.minimum(df["close"], df["open"]) - df["low"]) / candle_range

    # 5. Momentum Oscillators: RSI & Stochastic
    df["rsi_7"] = compute_rsi(df["close"], window=7)
    df["rsi_14"] = compute_rsi(df["close"], window=14)

    lowest_14 = df["low"].rolling(14).min()
    highest_14 = df["high"].rolling(14).max()
    stoch_k = ((df["close"] - lowest_14) / (highest_14 - lowest_14 + 1e-9)) * 100.0
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()

    # 6. Bollinger Bands
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    df["bb_pct_b"] = (df["close"] - lower_bb) / (upper_bb - lower_bb + 1e-9)
    df["bb_bandwidth"] = ((upper_bb - lower_bb) / sma_20) * 100.0

    # 7. Normalized Average True Range (NATR)
    df["natr_14"] = compute_atr(df["high"], df["low"], df["close"], window=14)

    # 8. Calendar features
    df["day_of_week"] = df["price_date"].dt.dayofweek
    df["day_of_month"] = df["price_date"].dt.day
    df["month"] = df["price_date"].dt.month
    df["is_month_end"] = df["price_date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["price_date"].dt.is_quarter_end.astype(int)

    # 9. Merge Constituent Banking Stock Features
    if os.path.exists(raw_constituents_path):
        const_df = pd.read_csv(raw_constituents_path)
        const_df["price_date"] = pd.to_datetime(const_df["price_date"])
        const_features = build_constituent_features(const_df)
        const_features["price_date"] = pd.to_datetime(const_features["price_date"])
        df = pd.merge(df, const_features, on="price_date", how="left")

    # 10. TARGET VARIABLES (Shifted by -1: Next Trading Session Return)
    # Target Return: percentage change from Close_t to Close_{t+1}
    df["target_return_next_1d"] = (df["close"].shift(-1) / df["close"] - 1.0) * 100.0
    # Direction: 1 if positive return, 0 otherwise
    df["target_direction_next_1d"] = (df["target_return_next_1d"] > 0).astype(int)

    # Drop warm-up rows (from 200-day SMA) and the last row (target is NaN)
    clean_df = df.dropna().reset_index(drop=True)
    return clean_df


def prepare_datasets(test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Run full feature engineering and perform chronological train-test split.
    Saves processed datasets to data/processed/.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_idx = os.path.join(RAW_DIR, "bank_nifty_raw.csv")
    raw_const = os.path.join(RAW_DIR, "bank_nifty_constituents.csv")

    full_df = engineer_bank_nifty_features(raw_idx, raw_const)
    full_path = os.path.join(PROCESSED_DIR, "bank_nifty_engineered.csv")
    full_df.to_csv(full_path, index=False)
    print(f"Full engineered dataset saved to {full_path} ({full_df.shape[0]} rows, {full_df.shape[1]} cols).")

    # Identify feature columns (exclude identifiers and targets)
    non_features = [
        "symbol", "stock", "ticker", "price_date",
        "open", "high", "low", "close", "volume", "dividends",
        "target_return_next_1d", "target_direction_next_1d"
    ]
    feature_cols = [c for c in full_df.columns if c not in non_features]

    # Chronological Split (strictly chronological to respect time-series nature)
    split_idx = int(len(full_df) * (1 - test_ratio))
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    test_df = full_df.iloc[split_idx:].reset_index(drop=True)

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train split: {len(train_df)} rows ({train_df['price_date'].min().strftime('%Y-%m-%d')} to {train_df['price_date'].max().strftime('%Y-%m-%d')})")
    print(f"Test split:  {len(test_df)} rows ({test_df['price_date'].min().strftime('%Y-%m-%d')} to {test_df['price_date'].max().strftime('%Y-%m-%d')})")
    print(f"Number of engineered features: {len(feature_cols)}")

    return train_df, test_df, feature_cols


if __name__ == "__main__":
    prepare_datasets()

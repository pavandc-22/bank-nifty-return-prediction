"""
Data ingestion module for Bank Nifty return prediction.
Fetches historical market data for the Bank Nifty index (^NSEBANK)
and its key constituent banking stocks using Yahoo Finance (yfinance).
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Target index and constituents configuration
INDEX_TICKER = "^NSEBANK"
INDEX_NAME = "Nifty Bank Index"
INDEX_SYMBOL = "BANKNIFTY"

CONSTITUENTS = [
    {"symbol": "HDFCBANK", "ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd."},
    {"symbol": "ICICIBANK", "ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd."},
    {"symbol": "SBIN", "ticker": "SBIN.NS", "name": "State Bank of India"},
    {"symbol": "KOTAKBANK", "ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd."},
    {"symbol": "AXISBANK", "ticker": "AXISBANK.NS", "name": "Axis Bank Ltd."},
    {"symbol": "INDUSINDBK", "ticker": "INDUSINDBK.NS", "name": "IndusInd Bank Ltd."},
    {"symbol": "FEDERALBNK", "ticker": "FEDERALBNK.NS", "name": "The Federal Bank Ltd."},
    {"symbol": "BANKBARODA", "ticker": "BANKBARODA.NS", "name": "Bank of Baroda"},
    {"symbol": "PNB", "ticker": "PNB.NS", "name": "Punjab National Bank"},
]


def fetch_ticker_data(ticker: str, symbol: str, stock_name: str, period: str = "7y") -> pd.DataFrame:
    """Fetch historical OHLCV + dividend data for a single ticker."""
    print(f"Fetching data for {symbol} ({ticker}) for period={period}...")
    obj = yf.Ticker(ticker)
    df = obj.history(period=period, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # Standardize column names
    df = df.reset_index()
    
    # Handle timezone if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        date_col = "Date"
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
        date_col = "Datetime"
    else:
        date_col = df.columns[0]

    df = df.rename(columns={
        date_col: "price_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Dividends": "dividends",
    })

    df["symbol"] = symbol
    df["stock"] = stock_name
    df["ticker"] = ticker

    # Select standard columns as per data description
    keep_cols = [
        "symbol", "stock", "ticker", "price_date",
        "open", "high", "low", "close", "volume", "dividends"
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols]

    # Clean numeric fields
    for col in ["open", "high", "low", "close", "volume", "dividends"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).sort_values("price_date").reset_index(drop=True)
    return df


def download_and_save_all(output_dir: str = DEFAULT_RAW_DIR, period: str = "7y"):
    """Download Bank Nifty index and all constituents, saving raw CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Download Bank Nifty Index
    print("=== Downloading Bank Nifty Index Data ===")
    index_df = fetch_ticker_data(
        ticker=INDEX_TICKER,
        symbol=INDEX_SYMBOL,
        stock_name=INDEX_NAME,
        period=period
    )
    index_path = os.path.join(output_dir, "bank_nifty_raw.csv")
    index_df.to_csv(index_path, index=False)
    print(f"Bank Nifty index data saved to {index_path} ({len(index_df)} rows).")

    # 2. Download constituents
    print("\n=== Downloading Constituent Bank Stocks ===")
    constituent_dfs = []
    for item in CONSTITUENTS:
        try:
            cdf = fetch_ticker_data(
                ticker=item["ticker"],
                symbol=item["symbol"],
                stock_name=item["name"],
                period=period
            )
            constituent_dfs.append(cdf)
        except Exception as e:
            print(f"Failed to fetch {item['symbol']}: {e}")

    if constituent_dfs:
        all_constituents_df = pd.concat(constituent_dfs, ignore_index=True)
        constituents_path = os.path.join(output_dir, "bank_nifty_constituents.csv")
        all_constituents_df.to_csv(constituents_path, index=False)
        print(f"Constituents data saved to {constituents_path} ({len(all_constituents_df)} rows).")

    return index_df, all_constituents_df


if __name__ == "__main__":
    download_and_save_all()

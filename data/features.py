from __future__ import annotations

import pandas as pd


def add_price_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OHLC-based distance indicators to the frame."""

    df = df.copy()
    df["oc_dist"] = df["close"] - df["open"]
    df["oh_dist"] = df["high"] - df["open"]
    df["hl_dist"] = df["high"] - df["low"]
    df["lc_dist"] = df["close"] - df["low"]
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_moving_averages(df: pd.DataFrame, short: int = 3, long: int = 12) -> pd.DataFrame:
    df = df.copy()
    df["MA3"] = df["close"].rolling(window=short, min_periods=short).mean()
    df["MA12"] = df["close"].rolling(window=long, min_periods=long).mean()
    df["MA_diff"] = df["MA3"] - df["MA12"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 12, num_std: float = 1.8) -> pd.DataFrame:
    df = df.copy()
    df["boll_std"] = df["close"].rolling(window=window, min_periods=window).std()
    df["boll_upper"] = df["MA12"] + num_std * df["boll_std"]
    df["boll_lower"] = df["MA12"] - num_std * df["boll_std"]
    df["boll_bandwidth"] = (df["boll_upper"] - df["boll_lower"]) / df["MA12"]
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe enriched with all technical indicators."""

    feature_df = add_price_distance_features(df)
    feature_df = add_rsi(feature_df)
    feature_df = add_moving_averages(feature_df)
    feature_df = add_bollinger_bands(feature_df)
    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df

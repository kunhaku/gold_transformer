from __future__ import annotations

import sqlite3
from typing import Protocol

import pandas as pd

from configs import DataConfig


class SupportsSqlConnection(Protocol):
    def cursor(self): ...


def load_mt5_data(config: DataConfig) -> pd.DataFrame:
    """Load OHLCV data from the configured SQLite database."""

    conn = sqlite3.connect(config.db_path)
    try:
        query = (
            f"SELECT time, open, high, low, close, tick_volume "
            f"FROM {config.table_name} ORDER BY time ASC"
        )
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    df["time"] = pd.to_datetime(df["time"])
    return df

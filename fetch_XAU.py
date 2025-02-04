import MetaTrader5 as mt5
import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime, timedelta

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 設定 SQLite 資料庫
DB_PATH = "mt5_data.db"
TABLE_NAME = "XAUUSD"
CSV_DIR = "data"
CSV_PATH = os.path.join(CSV_DIR, "XAUUSD_data.csv")


def initialize_mt5():
    """初始化 MetaTrader 5"""
    login = 6360731
    server = "OANDA-Demo-1"
    password = "Av!7UqPh"

    if not mt5.initialize(login=login, server=server, password=password):
        logging.error(f"initialize() failed, error code = {mt5.last_error()}")
        quit()
    logging.info("MT5 初始化成功。")


def fetch_data(symbol, timeframe, start_date, end_date):
    """抓取歷史數據"""
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logging.warning("未抓取到任何數據。")
            return None
        return pd.DataFrame(rates)
    except Exception as e:
        logging.error(f"抓取數據時發生錯誤: {e}")
        return None


def fetch_master_data(symbol, timeframe, log_func):
    """抓取 300 天的數據"""
    log_func("開始抓取 300 天的數據...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)
    df = fetch_data(symbol, timeframe, start_date, end_date)
    if df is None or df.empty:
        log_func("未抓取到數據。")
        return None

    # 檢查必要的欄位
    required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    if not all(col in df.columns for col in required_columns):
        log_func("數據缺少必要的欄位。")
        return None

    # 轉換時間欄位
    df['time'] = pd.to_datetime(df['time'], unit='s')
    log_func("成功抓取 300 天的數據。")
    return df


def save_to_sqlite(df):
    """將數據存入 SQLite 資料庫"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 建立表格（如果不存在）
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            time TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            tick_volume INTEGER
        )
    """)

    # 使用副本來轉換時間格式，避免影響原始 DataFrame
    df_copy = df.copy()
    df_copy['time'] = df_copy['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 插入數據
    df_copy.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()
    logging.info(f"數據已存入 SQLite: {DB_PATH}")


def save_to_csv(df):
    """將數據存入 CSV"""
    os.makedirs(CSV_DIR, exist_ok=True)

    # 確保時間格式轉換成字串 (ISO 8601 格式)
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df.to_csv(CSV_PATH, index=False)
    logging.info(f"數據已輸出為 CSV: {CSV_PATH}")


# 主函數
if __name__ == '__main__':
    # 設定 symbol 和 timeframe
    symbol = "XAUUSD.sml"
    timeframe = mt5.TIMEFRAME_M5

    # 初始化 MT5
    initialize_mt5()

    # 抓取數據，傳遞 logging.info 作為 log_func
    data = fetch_master_data(symbol, timeframe, logging.info)

    if data is not None:
        print("數據預覽：")
        print(data.head())

        # 存入 SQLite
        save_to_sqlite(data)

        # 存入 CSV
        save_to_csv(data)
    else:
        print("未抓取到數據。")

    # 關閉 MT5
    mt5.shutdown()

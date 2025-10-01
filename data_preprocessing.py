import sqlite3
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------- 1. 參數配置 --------------------
config = {
    "db_path": "mt5_data.db",        # SQLite 數據庫路徑
    "table_name": "main.XAUUSD",      # 數據表名稱
    "initial_input_length": 36,       # 最短的輸入長度
    "max_input_length": 48,           # 最大的輸入長度 (padding 用)
    "initial_forecast_length": 24,    # 最初的預測長度
    "min_forecast_length": 10,         # 最小的預測長度
    "num_samples_to_visualize": 5,    # 可視化的樣本數
    "train_ratio": 0.8,               # 訓練集佔比 (群組層面)
    # 定義原始特徵與新增的指標（依 config 順序決定最終的特徵排列）
    "base_features": ["open", "high", "low", "close", "tick_volume"],
    "additional_indicators": [
        "oc_dist", "oh_dist", "hl_dist", "lc_dist",
        "RSI", "MA3", "MA12", "MA_diff",
        "boll_upper", "boll_lower", "boll_bandwidth"
    ]
}


# -------------------- 2. 讀取數據並計算指標 --------------------
def load_mt5_data(db_path, table_name):
    """
    從 SQLite 數據庫讀取 OHLCV 數據，轉換為 Pandas DataFrame，
    並計算額外指標（RSI、MA、布林帶等）。
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT time, open, high, low, close, tick_volume FROM {table_name} ORDER BY time ASC"
    df = pd.read_sql(query, conn)
    conn.close()

    # 轉換時間格式
    df['time'] = pd.to_datetime(df['time'])

    # 計算 K 棒距離指標（保留原有）
    df['oc_dist'] = df['close'] - df['open']
    df['oh_dist'] = df['high'] - df['open']
    df['hl_dist'] = df['high'] - df['low']
    df['lc_dist'] = df['close'] - df['low']

    # RSI (採用 window=14)
    window = 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MA 指標：MA3 與 MA12，並計算差值
    df['MA3'] = df['close'].rolling(window=3, min_periods=3).mean()
    df['MA12'] = df['close'].rolling(window=12, min_periods=12).mean()
    df['MA_diff'] = df['MA3'] - df['MA12']

    # 布林帶：以 MA12 為中軌，採用1.8標準差計算上軌與下軌
    df['boll_std'] = df['close'].rolling(window=12, min_periods=12).std()
    df['boll_upper'] = df['MA12'] + 1.8 * df['boll_std']
    df['boll_lower'] = df['MA12'] - 1.8 * df['boll_std']
    # 布林帶寬度（可歸一化）
    df['boll_bandwidth'] = (df['boll_upper'] - df['boll_lower']) / df['MA12']

    # 移除因 rolling 計算而產生的 NaN
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# -------------------- 3. 生成資料 + 分群 + 分割 --------------------
def prepare_data_and_split(df, config):
    """
    1) 按照動態窗口產生 (X, y, mask)
    2) 給每筆樣本一個 group_id (並產生 y_mask)
    3) 若最後一組未完成，則排除該群組
    4) 依 group_id 切分 train/test
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # 依 config 定義最終特徵順序
    features = config["base_features"] + config["additional_indicators"]

    X_list, y_list, mask_list, group_id_list = [], [], [], []
    y_mask_list = []

    # 取出指定欄位，注意欄位順序必須與 config["features"] 一致
    data = df[features].values  # shape = (total_length, num_features)
    total_length = len(data)

    input_length = config["initial_input_length"]
    forecast_length = config["initial_forecast_length"]
    max_input_length = config["max_input_length"]
    min_forecast_length = config["min_forecast_length"]

    start_index = 0
    current_group_id = 0
    last_completed_group_id = -1

    while start_index + input_length + forecast_length <= total_length:
        X = data[start_index: start_index + input_length]
        # 以 close 為目標，close 在 features 裡面位於第 4 (若從0計算 index=3)
        y = data[start_index + input_length: start_index + input_length + forecast_length, 3]
        mask = np.ones((input_length, 1))
        y_mask_unpadded = np.ones((forecast_length,), dtype='float32')

        X_list.append(X)
        y_list.append(y)
        mask_list.append(mask)
        group_id_list.append(current_group_id)
        y_mask_list.append(y_mask_unpadded)

        if forecast_length > min_forecast_length:
            input_length += 1
            forecast_length -= 1
        else:
            last_completed_group_id = current_group_id
            current_group_id += 1
            input_length = config["initial_input_length"]
            forecast_length = config["initial_forecast_length"]
            start_index += config["initial_input_length"]

        start_index += 1

    filtered_X_list, filtered_y_list, filtered_mask_list, filtered_group_id_list, filtered_y_mask_list = [], [], [], [], []
    for X_item, y_item, m_item, gid_item, y_mask_item in zip(X_list, y_list, mask_list, group_id_list, y_mask_list):
        if gid_item <= last_completed_group_id:
            filtered_X_list.append(X_item)
            filtered_y_list.append(y_item)
            filtered_mask_list.append(m_item)
            filtered_group_id_list.append(gid_item)
            filtered_y_mask_list.append(y_mask_item)

    X_padded = pad_sequences(filtered_X_list, maxlen=max_input_length, dtype='float32', padding='pre', value=0.0)
    mask_padded = pad_sequences(filtered_mask_list, maxlen=max_input_length, dtype='float32', padding='pre', value=0.0)
    max_forecast_length = config["initial_forecast_length"]
    y_padded = pad_sequences(filtered_y_list, maxlen=max_forecast_length, dtype='float32', padding='post', value=0.0)
    y_mask_padded = pad_sequences(filtered_y_mask_list, maxlen=max_forecast_length, dtype='float32', padding='post', value=0.0)
    group_ids = np.array(filtered_group_id_list, dtype=np.int32)

    unique_groups = np.unique(group_ids)
    num_groups = len(unique_groups)
    cut_idx = int(num_groups * config["train_ratio"])

    train_groups = unique_groups[:cut_idx]
    test_groups = unique_groups[cut_idx:]

    train_mask = np.isin(group_ids, train_groups)
    test_mask = np.isin(group_ids, test_groups)

    X_train = X_padded[train_mask]
    y_train = y_padded[train_mask]
    mask_train = mask_padded[train_mask]
    y_mask_train = y_mask_padded[train_mask]
    group_train = group_ids[train_mask]

    X_test = X_padded[test_mask]
    y_test = y_padded[test_mask]
    mask_test = mask_padded[test_mask]
    y_mask_test = y_mask_padded[test_mask]
    group_test = group_ids[test_mask]

    return (X_train, y_train, mask_train, y_mask_train, group_train), (X_test, y_test, mask_test, y_mask_test, group_test)


# -------------------- 4. 可視化 (簡易打印) --------------------
def visualize_data(X, y, mask, group_ids, num_samples=3):
    num_samples = min(num_samples, X.shape[0])
    for i in range(num_samples):
        gid = group_ids[i]
        input_length = int(mask[i].sum())
        print(f"\n📌 樣本 {i+1} (Group={gid}, 有效輸入長度: {input_length})")
        input_values = X[i, :, 3].astype(int)  # 這裡顯示 close 值
        output_values = y[i].astype(int)
        print("🔹 輸入:", input_values.tolist())
        print("🔹 輸出:", output_values.tolist())


def save_data_numpy(X_data, y_data, mask_data, y_mask_data, group_ids, save_path):
    np.savez_compressed(save_path,
                        X_data=X_data,
                        y_data=y_data,
                        mask_data=mask_data,
                        y_mask_data=y_mask_data,
                        group_ids=group_ids)
    print(f"✅ 已儲存至 {save_path}")


# -------------------- 5. 主程式 --------------------
if __name__ == "__main__":
    df = load_mt5_data(config["db_path"], config["table_name"])
    print("原始資料筆數:", len(df))
    (X_train, y_train, mask_train, y_mask_train, g_train), (X_test, y_test, mask_test, y_mask_test, g_test) = prepare_data_and_split(df, config)
    print("\n[Train] X shape:", X_train.shape, "y shape:", y_train.shape, "mask shape:", mask_train.shape)
    print("[Test]  X shape:", X_test.shape, "y shape:", y_test.shape, "mask shape:", mask_test.shape)
    print("Train group數:", len(np.unique(g_train)))
    print("Test group數:", len(np.unique(g_test)))
    print("\n=== Train集(前幾筆) ===")
    visualize_data(X_train, y_train, mask_train, g_train, num_samples=config["num_samples_to_visualize"])
    save_data_numpy(X_train, y_train, mask_train, y_mask_train, g_train, "train_raw.npz")
    save_data_numpy(X_test, y_test, mask_test, y_mask_test, g_test, "test_raw.npz")
    print("全部處理完畢!")

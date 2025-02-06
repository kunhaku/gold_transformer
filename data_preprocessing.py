import sqlite3
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------- 1. 參數配置 --------------------
config = {
    "db_path": "mt5_data.db",        # SQLite 數據庫路徑
    "table_name": "main.XAUUSD",     # 數據表名稱
    "initial_input_length": 20,      # 最短的輸入長度
    "max_input_length": 30,          # 最大的輸入長度 (padding 用)
    "initial_forecast_length": 10,   # 最初的預測長度
    "min_forecast_length": 3,        # 最小的預測長度
    "num_samples_to_visualize": 5,   # 可視化的樣本數
    "train_ratio": 0.8               # 訓練集佔比 (群組層面)
}


# -------------------- 2. 讀取數據 --------------------
def load_mt5_data(db_path, table_name):
    """
    從 SQLite 數據庫讀取 OHLCV 數據，轉換為 Pandas DataFrame。
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT time, open, high, low, close, tick_volume FROM {table_name} ORDER BY time ASC"
    df = pd.read_sql(query, conn)
    conn.close()

    # 轉換時間格式
    df['time'] = pd.to_datetime(df['time'])
    return df


# -------------------- 3. 生成資料 + 分群 + 分割 --------------------
def prepare_data_and_split(df, config):
    """
    1) 按照動態窗口產生 (X, y, mask)
    2) 給每筆樣本一個 group_id
    3) 若最後一組未完成 (尚未達到結束條件)，則將該未完成的群組排除
    4) 最後依 group_id 切分 train/test
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X_list, y_list, mask_list, group_id_list = [], [], [], []
    y_mask_list = []

    data = df[['open', 'high', 'low', 'close', 'tick_volume']].values
    total_length = len(data)

    input_length = config["initial_input_length"]
    forecast_length = config["initial_forecast_length"]
    max_input_length = config["max_input_length"]
    min_forecast_length = config["min_forecast_length"]

    start_index = 0
    current_group_id = 0  # 從 0 開始編群
    last_completed_group_id = -1  # *** 新增：用來紀錄最後結束的群組

    while start_index + input_length + forecast_length <= total_length:
        X = data[start_index: start_index + input_length]
        y = data[start_index + input_length: start_index + input_length + forecast_length, 3]  # close
        mask = np.ones((input_length, 1))

        # - 產生 y_mask: shape=(forecast_length,)，都為 1
        #   後續會再 pad 到 (max_forecast_length,)。
        #   當然，也可以先建立成 shape=(max_forecast_length,)，前f個1、後面0
        #   看你 padding 方式而定。這裡先產生 shape=(f,)
        y_mask_unpadded = np.ones((forecast_length,), dtype='float32')

        X_list.append(X)
        y_list.append(y)
        mask_list.append(mask)
        group_id_list.append(current_group_id)
        # 先存下這個 unpadded mask
        y_mask_list.append(y_mask_unpadded)

        # 動態調整 input_length/forecast_length
        if forecast_length > min_forecast_length:
            input_length += 1
            forecast_length -= 1
        else:
            # 代表這個 group 已完成，開始新的一組
            last_completed_group_id = current_group_id  # *** 新增：成功結束群組後更新
            current_group_id += 1
            # 重置輸入長度/預測長度
            input_length = config["initial_input_length"]
            forecast_length = config["initial_forecast_length"]
            # 跳過一定區間
            start_index += config["initial_input_length"]

        start_index += 1

    # *** 新增：過濾掉尚未完成的 group
    filtered_X_list = []
    filtered_y_list = []
    filtered_mask_list = []
    filtered_group_id_list = []
    filtered_y_mask_list = []

    for X_item, y_item, m_item, gid_item, y_mask_item in zip(X_list, y_list, mask_list, group_id_list, y_mask_list):
        if gid_item <= last_completed_group_id:
            filtered_X_list.append(X_item)
            filtered_y_list.append(y_item)
            filtered_mask_list.append(m_item)
            filtered_group_id_list.append(gid_item)
            filtered_y_mask_list.append(y_mask_item)

    # === pad X & mask ===
    X_padded = pad_sequences(filtered_X_list, maxlen=max_input_length, dtype='float32', padding='pre', value=0.0)
    mask_padded = pad_sequences(filtered_mask_list, maxlen=max_input_length, dtype='float32', padding='pre', value=0.0)

    # === pad y (跟你原本相同) ===

    max_forecast_length = config["initial_forecast_length"]
    y_padded = pad_sequences(filtered_y_list, maxlen=max_forecast_length, dtype='float32', padding='post', value=0.0)

    # === pad y_mask (跟 y 一樣的長度) ===
    y_mask_padded = pad_sequences(filtered_y_mask_list, maxlen=max_forecast_length, dtype='float32', padding='post',
                                  value=0.0)
    # y_mask_padded.shape = (num_samples, max_forecast_length)

    group_ids = np.array(filtered_group_id_list, dtype=np.int32)

    # 依 group_id 分割 train/test
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
    group_train = group_ids[train_mask]

    X_test = X_padded[test_mask]
    y_test = y_padded[test_mask]
    mask_test = mask_padded[test_mask]
    group_test = group_ids[test_mask]

    # train
    y_mask_train = y_mask_padded[train_mask]
    ...
    # test
    y_mask_test = y_mask_padded[test_mask]


    return (X_train, y_train, mask_train, y_mask_train, group_train), (X_test, y_test, mask_test, y_mask_test, group_test)




# -------------------- 4. 可視化 (簡易打印) --------------------
def visualize_data(X, y, mask, group_ids, num_samples=3):
    num_samples = min(num_samples, X.shape[0])
    for i in range(num_samples):
        gid = group_ids[i]
        input_length = int(mask[i].sum())  # 計算有效輸入長度

        print(f"\n📌 **樣本 {i + 1}** (Group={gid}, 有效輸入長度: {input_length})")
        # 只打印 Close (X[...,3])
        input_values = X[i, :, 3].astype(int)
        output_values = y[i].astype(int)
        print("🔹 輸入 (部份):", input_values.tolist())
        print("🔹 輸出:", output_values.tolist())


def save_data_numpy(X_data, y_data, mask_data, y_mask_data, group_ids, save_path):
    np.savez_compressed(
        save_path,
        X_data=X_data,
        y_data=y_data,
        mask_data=mask_data,
        y_mask_data=y_mask_data,
        group_ids=group_ids
    )
    print(f"✅ 已儲存至 {save_path}")



# -------------------- 5. 主程式 --------------------
if __name__ == "__main__":
    # 讀取原始數據
    df = load_mt5_data(config["db_path"], config["table_name"])
    print("原始資料筆數:", len(df))

    # 產生 (X, y, mask) + 分群 + train/test split
    (X_train, y_train, m_train,y_mask_train, g_train), (X_test, y_test, m_test, y_mask_test, g_test) = prepare_data_and_split(df, config)

    print("\n[Train] X shape:", X_train.shape, "y shape:", y_train.shape, "mask shape:", m_train.shape)
    print("[Test]  X shape:", X_test.shape,  "y shape:", y_test.shape,  "mask shape:", m_test.shape)
    print("Train group數:", len(np.unique(g_train)))
    print("Test group數:", len(np.unique(g_test)))

    # 簡單檢查
    print("\n=== Train集(前幾筆) ===")
    visualize_data(X_train, y_train, m_train, g_train, num_samples=config["num_samples_to_visualize"])

    # 分別儲存
    save_data_numpy(X_train, y_train, m_train, y_mask_train, g_train, "train_raw.npz")
    save_data_numpy(X_test, y_test, m_test, y_mask_test, g_test, "test_raw.npz")

    print("全部處理完畢!")

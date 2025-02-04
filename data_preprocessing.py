import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

### -------------------- 1. 參數配置 --------------------
config = {
    "db_path": "mt5_data.db",  # SQLite 數據庫路徑
    "table_name": "main.XAUUSD",  # 數據表名稱
    "initial_input_length": 20,  # 最短的輸入長度
    "max_input_length": 30,  # 最大的輸入長度
    "initial_forecast_length": 10,  # 最初的預測長度
    "min_forecast_length": 3,  # 最小的預測長度
    "num_samples_to_visualize": 10  # 可視化的樣本數
}


### -------------------- 2. 讀取數據 --------------------
def load_mt5_data(db_path, table_name=config["table_name"]):
    """
    從 SQLite 數據庫讀取 OHLCV 數據，轉換為 Pandas DataFrame。

    Args:
        db_path (str): 數據庫文件路徑
        table_name (str): OHLCV 數據表名稱

    Returns:
        pd.DataFrame: 轉換後的 OHLCV 數據
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT time, open, high, low, close, tick_volume FROM {table_name} ORDER BY time ASC"
    df = pd.read_sql(query, conn)
    conn.close()

    # 轉換時間格式
    df['time'] = pd.to_datetime(df['time'])
    return df


### -------------------- 3. 預處理數據 --------------------
def prepare_data(df, config):
    """
    生成訓練數據，動態變化輸入長度與預測長度，並應用 Masking。
    當預測長度 (`forecast_length`) 達到 `min_forecast_length`，將 `input_length` 重置並開始新的滑動窗口。

    Args:
        df (pd.DataFrame): 包含 OHLCV 數據的 DataFrame。
        config (dict): 超參數字典。

    Returns:
        X_data: 形狀為 (num_samples, max_input_length, 5) 的輸入數據 (填充)。
        y_data: 形狀為 (num_samples, max_forecast_length) 的目標數據。
        mask_data: 形狀為 (num_samples, max_input_length, 1) 的 Masking 矩陣。
    """
    X_list, y_list, mask_list = [], [], []

    data = df[['open', 'high', 'low', 'close', 'tick_volume']].values
    total_length = len(data)

    # 讀取參數
    input_length = config["initial_input_length"]
    forecast_length = config["initial_forecast_length"]
    max_input_length = config["max_input_length"]
    min_forecast_length = config["min_forecast_length"]
    max_forecast_length = forecast_length  # 記錄最大預測長度

    start_index = 0

    while start_index + input_length + forecast_length <= total_length:
        X = data[start_index:start_index + input_length]  # 取得輸入數據
        y = data[start_index + input_length : start_index + input_length + forecast_length, 3]  # 取得完整預測序列

        # 生成 Mask
        mask = np.ones((input_length, 1))

        X_list.append(X)
        y_list.append(y)
        mask_list.append(mask)

        # 調整輸入長度 & 預測長度
        if forecast_length > min_forecast_length:
            input_length += 1  # 增加輸入長度
            forecast_length -= 1  # 預測長度減少
        else:
            input_length = config["initial_input_length"]  # 重置輸入長度
            forecast_length = config["initial_forecast_length"]  # 重置預測長度
            start_index += config["initial_input_length"]  # 跳過一定區間，避免過度重疊

        start_index += 1  # 滑動窗口前移

    # 使用 pad_sequences 進行填充
    X_padded = pad_sequences(X_list, maxlen=max_input_length, dtype='float32', padding='pre', value=0.0)
    mask_padded = pad_sequences(mask_list, maxlen=max_input_length, dtype='float32', padding='pre', value=0.0)
    y_padded = pad_sequences(y_list, maxlen=max_forecast_length, dtype='float32', padding='post', value=0.0)

    return X_padded, y_padded, mask_padded



### -------------------- 4. 可視化數據 --------------------
import matplotlib.pyplot as plt
import numpy as np


def visualize_data(X, y, mask, num_samples=3):
    """
    以數字陣列方式顯示前 num_samples 組數據，包括:
    - 輸入 (X)
    - 預測 (y)

    Args:
        X (np.array): 預處理後的輸入數據。
        y (np.array): 預測目標 Close 價格 (完整預測序列)。
        mask (np.array): Masking 矩陣。
        num_samples (int): 顯示的樣本數 (預設前3組)。
    """
    num_samples = min(num_samples, X.shape[0])  # 確保不超過資料總數

    print("\n=== 資料可視化 (數字格式) ===")

    for i in range(num_samples):
        input_length = int(mask[i].sum())  # 計算有效輸入長度
        print(f"\n📌 **樣本 {i + 1}** (輸入長度: {input_length})")

        # 取得輸入數據 (只取 close 價格，並對齊 padding)
        input_values = X[i, :, 3].astype(int)  # 取 Close 價格並轉換為整數方便閱讀
        print("🔹 **輸入:**")
        print(input_values.tolist())

        # 取得預測數據 (對齊 padding)
        output_values = y[i, :].astype(int)  # 轉換為整數
        print("🔹 **輸出:**")
        print(output_values.tolist())

    print("\n=== 顯示結束 ===\n")

def save_data_numpy(X_data, y_data, mask_data, save_path="dataset.npz"):
    """
    以 NumPy `.npz` 格式儲存數據，方便 TensorFlow 讀取。

    Args:
        X_data (np.array): 輸入數據。
        y_data (np.array): 預測目標數據。
        mask_data (np.array): Masking 矩陣。
        save_path (str): 儲存的文件名稱。
    """
    np.savez_compressed(save_path, X_data=X_data, y_data=y_data, mask_data=mask_data)
    print(f"✅ 數據已成功儲存至 {save_path}")


### -------------------- 5. 主程式 --------------------
if __name__ == "__main__":
    # 讀取數據
    df = load_mt5_data(config["db_path"], config["table_name"])

    # 預處理數據
    X_data, y_data, mask_data = prepare_data(df, config)

    # 顯示數據形狀
    print(f"X_data shape: {X_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    print(f"mask_data shape: {mask_data.shape}")

    # **更新: 使用新的可視化函數**
    visualize_data(X_data, y_data, mask_data, num_samples=config["num_samples_to_visualize"])
    # 執行儲存
    save_data_numpy(X_data, y_data, mask_data)


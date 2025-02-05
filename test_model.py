import os
import sqlite3
import numpy as np
import tensorflow as tf

def main():
    # === 1) 讀取測試資料 ===
    data_test = np.load("test_scaled.npz")
    X_test = data_test["X_data"]       # shape: (num_test_samples, seq_len, input_dim)
    y_test = data_test["y_data"]       # shape: (num_test_samples, forecast_length)
    m_test = data_test["mask_data"]    # shape: (num_test_samples, seq_len, 1)
    g_test = data_test["group_ids"]    # shape: (num_test_samples,)

    forecast_length = y_test.shape[1]
    input_dim = X_test.shape[2]
    unique_test_groups = np.unique(g_test)

    print("Test X shape:", X_test.shape)
    print("Test y shape:", y_test.shape)
    print("Test mask shape:", m_test.shape)
    print("Test group_ids shape:", g_test.shape)
    print("Test group 數量:", len(unique_test_groups))

    # === 2) 載入訓練好的模型 (model_b) ===
    model_path = r"G:\GoldFX_Transformer\models\model_b"
    print(f"載入模型: {model_path}")
    # 如果之前的模型是以 subclass Model 寫的，需要用 custom_objects 或 compile=False
    model = tf.keras.models.load_model(model_path, compile=False)
    # 也可以 model.compile() 看你是否需要 metrics / loss

    # === 3) 建立 / 連線 SQLite 資料庫, 並建表 ===
    #    這裡示範寫在同目錄下 "predictions.db"，表名為 "model_b_inference"
    db_path = "predictions.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 依照 forecast_length 動態生成資料表欄位: y_1..y_n, p_1..p_n
    # 例如 forecast_length=10 -> y_1 float, ..., y_10 float, p_1 float, ..., p_10 float
    y_cols = ", ".join([f"y_{i+1} float" for i in range(forecast_length)])
    p_cols = ", ".join([f"p_{i+1} float" for i in range(forecast_length)])
    # 組合成 CREATE TABLE 語句
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS model_b_inference (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INT,
        sample_idx INT,
        {y_cols},
        {p_cols}
    )
    """
    cursor.execute(create_table_sql)

    # 我們需要依照欄位數量動態寫 INSERT 語句
    # 前面固定兩個欄位: group_id, sample_idx
    # 後面 2*forecast_length 個欄位: y_1..y_n, p_1..p_n
    placeholders = ", ".join(["?"] * (2 + forecast_length*2))
    insert_sql = f"INSERT INTO model_b_inference (group_id, sample_idx, {','.join([f'y_{i+1}' for i in range(forecast_length)])}, {','.join([f'p_{i+1}' for i in range(forecast_length)])}) VALUES ({placeholders})"

    print("資料表 model_b_inference 建立/確認完成，開始推論並寫入資料...")

    # === 4) 在測試集上逐 group 推論 ===
    #     與訓練類似：同一 group 內，使用 past_preds 逐步 refine
    for g in unique_test_groups:
        idxs = np.where(g_test == g)[0]
        # 為了模擬實際情況，每個 group 開始時 past_preds = 0
        past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

        for i in idxs:
            x_i = X_test[i][None, ...]  # (1, seq_len, input_dim)
            y_i = y_test[i][None, ...]  # (1, forecast_length)
            m_i = m_test[i][None, ...]  # (1, seq_len, 1)

            # 推論
            preds = model(x_i, mask=m_i, past_preds=past_preds, training=False)
            past_preds = preds  # 更新 past_preds

            # 取 numpy
            pred_values = preds.numpy().flatten()  # shape = (forecast_length,)
            true_values = y_i.flatten()            # shape = (forecast_length,)

            # 組合要寫入的 row: group_id, sample_idx, y_1..y_n, p_1..p_n
            row_data = [int(g), int(i)] + list(true_values) + list(pred_values)

            # 插入 DB
            cursor.execute(insert_sql, row_data)

    # 提交並關閉
    conn.commit()
    conn.close()
    print(f"推論結束，結果已寫入 SQLite: {db_path} -> [model_b_inference] 表")

if __name__ == "__main__":
    main()

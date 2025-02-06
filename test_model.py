import os
import sqlite3
import numpy as np
import tensorflow as tf

def main():
    # === 1) 讀取測試資料 ===
    data_test = np.load("test_raw.npz")
    X_test = data_test["X_data"]       # (num_test_samples, seq_len, input_dim)
    y_test = data_test["y_data"]       # (num_test_samples, forecast_length)
    m_test = data_test["mask_data"]    # (num_test_samples, seq_len, 1)
    y_mask_test = data_test["y_mask_data"]  # <--- 新增: (num_test_samples, forecast_length)
    g_test = data_test["group_ids"]    # (num_test_samples,)

    forecast_length = y_test.shape[1]
    input_dim = X_test.shape[2]
    unique_test_groups = np.unique(g_test)

    print("Test X shape:", X_test.shape)
    print("Test y shape:", y_test.shape)
    print("Test mask shape:", m_test.shape)
    print("Test y_mask shape:", y_mask_test.shape)
    print("Test group_ids shape:", g_test.shape)
    print("Test group 數量:", len(unique_test_groups))

    # === 2) 載入訓練好的模型 (model_b) ===
    model_path = r"G:\GoldFX_Transformer\models\model_b"
    print(f"載入模型: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # === 3) 建立 / 連線 SQLite 資料庫, 並建表 ===
    db_path = "predictions.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 依照 forecast_length 動態生成資料表欄位: y_1..y_n, p_1..p_n
    # 新增 valid_length 欄位用於記錄該樣本真正有效的預測長度
    y_cols = ", ".join([f"y_{i+1} REAL" for i in range(forecast_length)])
    p_cols = ", ".join([f"p_{i+1} REAL" for i in range(forecast_length)])
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS model_b_inference (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INT,
        sample_idx INT,
        valid_length INT,
        {y_cols},
        {p_cols}
    )
    """
    # 先刪除舊表，以確保新表的結構包含 valid_length 欄位
    cursor.execute("DROP TABLE IF EXISTS model_b_inference")
    cursor.execute(create_table_sql)

    # INSERT 語句: 多了 valid_length
    placeholders = ", ".join(["?"] * (3 + forecast_length*2))  # group_id, sample_idx, valid_length + y_1..y_n, p_1..p_n
    insert_sql = f"INSERT INTO model_b_inference (group_id, sample_idx, valid_length, {','.join([f'y_{i+1}' for i in range(forecast_length)])}, {','.join([f'p_{i+1}' for i in range(forecast_length)])}) VALUES ({placeholders})"

    print("資料表 model_b_inference 建立/確認完成，開始推論並寫入資料...")

    # === 4) 在測試集上逐 group 推論 ===
    for g in unique_test_groups:
        idxs = np.where(g_test == g)[0]
        past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

        for i in idxs:
            x_i = X_test[i][None, ...]  # (1, seq_len, input_dim)
            y_i = y_test[i][None, ...]  # (1, forecast_length)
            m_i = m_test[i][None, ...]  # (1, seq_len, 1)
            y_mask_i = y_mask_test[i][None, ...] # (1, forecast_length)

            # 計算該樣本的有效預測長度
            # sum(y_mask=1) => valid_length
            valid_len = int(np.sum(y_mask_i))

            # 推論
            preds = model(x_i, mask=m_i, past_preds=past_preds, training=False)
            past_preds = preds

            pred_values = preds.numpy().flatten()  # (forecast_length,)
            true_values = y_i.flatten()            # (forecast_length,)

            # 組合要寫入的 row: group_id, sample_idx, valid_length, y_1.., p_1..
            row_data = [
                int(g), int(i), valid_len
            ] + [float(v) for v in true_values] + [float(p) for p in pred_values]

            cursor.execute(insert_sql, row_data)

    # 提交並關閉
    conn.commit()
    conn.close()
    print(f"推論結束，結果已寫入 SQLite: {db_path} -> [model_b_inference] 表")


if __name__ == "__main__":
    main()

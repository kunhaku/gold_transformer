# config.py
config = {
    "db_path": "mt5_data.db",
    "table_name": "main.XAUUSD",
    "initial_input_length": 20,
    "max_input_length": 30,
    "initial_forecast_length": 10,
    "min_forecast_length": 3,
    "num_samples_to_visualize": 5,
    "train_ratio": 0.8,
    # 定義原始特徵與新增的指標：
    "base_features": ["open", "high", "low", "close", "tick_volume"],
    "additional_indicators": ["oc_dist", "oh_dist", "hl_dist", "lc_dist"]
}

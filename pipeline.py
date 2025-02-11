# pipeline.py
from config import config
from data_preprocessing import load_mt5_data, prepare_data_and_split, save_data_numpy
from raw_recurrent_transformer import run_training
from test_model import run_test_model
from visual_tool import run_visual_tool


def main():
    # 讀取數據
    df = load_mt5_data(config["db_path"], config["table_name"])

    # 預處理資料：產生 X, y, mask, y_mask, group_ids，並切分 train/test
    (X_train, y_train, mask_train, y_mask_train, group_train), \
        (X_test, y_test, mask_test, y_mask_test, group_test) = prepare_data_and_split(df, config)

    # 儲存處理後的資料（方便日後調試與檢查）
    save_data_numpy(X_train, y_train, mask_train, y_mask_train, group_train, "train_scaled.npz")
    save_data_numpy(X_test, y_test, mask_test, y_mask_test, group_test, "test_scaled.npz")

    # 訓練模型，傳入 train 資料與 config
    model = run_training(X_train, y_train, mask_train, y_mask_train, group_train, config)

    # 產生測試預測並儲存結果（例如存入 SQLite 或 npz）
    run_test_model(model, X_test, y_test, mask_test, y_mask_test, group_test, config)

    # 啟動視覺化工具（Dash App）
    run_visual_tool(config)


if __name__ == "__main__":
    main()

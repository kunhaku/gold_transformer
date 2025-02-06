import numpy as np
from sklearn.preprocessing import RobustScaler

def group_based_scaling(X_data, mask_data, group_ids):
    scaled_X_data = np.copy(X_data)
    scalers_dict = {}
    unique_g = np.unique(group_ids)

    for g in unique_g:
        idxs = np.where(group_ids == g)[0]
        if len(idxs) == 0:
            continue

        X_g = X_data[idxs]
        m_g = mask_data[idxs]

        # 收集該 group 的所有有效 time steps
        ohlc_list = []
        vol_list = []
        for i in range(len(X_g)):
            valid_len = int(m_g[i].sum())
            ohlc_list.append(X_g[i, 0:valid_len, 0:4])
            vol_list.append(X_g[i, 0:valid_len, 4:5])
        ohlc_concat = np.concatenate(ohlc_list, axis=0)
        vol_concat = np.concatenate(vol_list, axis=0)

        # 將 StandardScaler 改為 RobustScaler
        scaler_ohlc = RobustScaler()
        scaler_vol = RobustScaler()

        if len(ohlc_concat) > 0:
            scaler_ohlc.fit(ohlc_concat)
        if len(vol_concat) > 0:
            scaler_vol.fit(vol_concat)

        # 將資料 transform 回原位置
        for i in range(len(X_g)):
            valid_len = int(m_g[i].sum())
            scaled_ohlc = scaler_ohlc.transform(X_g[i, 0:valid_len, 0:4])
            scaled_vol  = scaler_vol.transform(X_g[i, 0:valid_len, 4:5])
            scaled_X_data[idxs[i], 0:valid_len, 0:4] = scaled_ohlc
            scaled_X_data[idxs[i], 0:valid_len, 4:5] = scaled_vol

        scalers_dict[g] = (scaler_ohlc, scaler_vol)

    return scaled_X_data, scalers_dict


def apply_group_scaling(X_data, mask_data, group_ids, scalers_dict):
    scaled_X_data = np.copy(X_data)
    unique_g = np.unique(group_ids)

    for g in unique_g:
        idxs = np.where(group_ids == g)[0]
        if len(idxs) == 0:
            continue

        if g not in scalers_dict:
            # 若該 group 未出現在訓練集，可視需求處理 (這裡示範直接略過)
            continue

        scaler_ohlc, scaler_vol = scalers_dict[g]
        X_g = X_data[idxs]
        m_g = mask_data[idxs]

        for i in range(len(X_g)):
            valid_len = int(m_g[i].sum())
            scaled_ohlc = scaler_ohlc.transform(X_g[i, 0:valid_len, 0:4])
            scaled_vol  = scaler_vol.transform(X_g[i, 0:valid_len, 4:5])
            scaled_X_data[idxs[i], 0:valid_len, 0:4] = scaled_ohlc
            scaled_X_data[idxs[i], 0:valid_len, 4:5] = scaled_vol

    return scaled_X_data


def main():
    # 1. 讀取 train_raw.npz
    train_raw = np.load("train_raw.npz")
    X_train = train_raw["X_data"]
    y_train = train_raw["y_data"]
    m_train = train_raw["mask_data"]
    g_train = train_raw["group_ids"]

    # (若原始資料中也包含 y_mask_data，就一起讀取)
    y_mask_train = train_raw["y_mask_data"]  # 可視實際需求讀取

    # 2. 對 train 的 X_data 做 group-based scaling
    scaled_X_train, scalers_dict = group_based_scaling(X_train, m_train, g_train)

    # 3. 儲存 scaled train 與不需要縮放的資料
    np.savez_compressed("train_scaled.npz",
                        X_data=scaled_X_train,   # 已縮放
                        y_data=y_train,         # 未縮放
                        mask_data=m_train,      # 未縮放
                        group_ids=g_train,       # 未縮放
                        y_mask_data=y_mask_train
    )
    print("已產生 train_scaled.npz")

    # 4. 讀取 test_raw.npz
    test_raw = np.load("test_raw.npz")
    X_test = test_raw["X_data"]
    y_test = test_raw["y_data"]
    m_test = test_raw["mask_data"]
    g_test = test_raw["group_ids"]

    # (若原始測試資料中也有 y_mask_test，同樣可一起讀取)
    y_mask_test = test_raw["y_mask_data"]

    # 5. 用前面 train 得到的 scalers_dict 對 test 的 X_data 做 transform
    scaled_X_test = apply_group_scaling(X_test, m_test, g_test, scalers_dict)

    # 6. 儲存 scaled test 與不需要縮放的資料
    np.savez_compressed("test_scaled.npz",
                        X_data=scaled_X_test,  # 已縮放
                        y_data=y_test,         # 未縮放
                        mask_data=m_test,      # 未縮放
                        group_ids=g_test,
                        y_mask_data=y_mask_test
    )
    print("已產生 test_scaled.npz")


if __name__ == "__main__":
    main()

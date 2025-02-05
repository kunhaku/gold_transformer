import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------------------------
# 1) 建立自訂的 Transformer Encoder (與你的程式類似)
# -------------------------------------------------------------------
class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, mask=None, training=False):
        """
        inputs: (batch_size, seq_len, embed_dim)
        mask:   (batch_size, seq_len, 1) 先轉成 MultiHeadAttention 需要的形狀 (batch, num_heads, seq_len, seq_len)
        """
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            mask = tf.squeeze(mask, axis=-1)        # (batch_size, seq_length)
            mask = tf.expand_dims(mask, axis=1)     # (batch_size, 1, seq_length)
            mask = tf.expand_dims(mask, axis=1)     # (
            # batch_size, 1, 1, seq_length)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])

        attn_output = self.attention(
            query=inputs, value=inputs, key=inputs,
            attention_mask=mask, training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# -------------------------------------------------------------------
# 2) 建立「可接受 past_preds」的模型
#    - 透過 subclass Model，將 past_preds 與 X_data 在 call() 中做 concat
# -------------------------------------------------------------------
class RecurrentTransformerModel(keras.Model):
    def __init__(self,
                 input_dim,               # X_data最後一維的特徵數量 (e.g. OHLCV=5)
                 forecast_length,         # 預測輸出長度 (e.g. 10)
                 num_layers=2,
                 embed_dim=64,
                 num_heads=4,
                 ff_dim=128,
                 dropout_rate=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.forecast_length = forecast_length

        # 將 (input_dim + forecast_length) -> embed_dim
        # 因為要在特徵維度 concat past_preds (batch, seq_len, forecast_length)
        # 所以實際輸入維度 = input_dim + forecast_length
        self.input_projection = Dense(embed_dim)

        self.encoders = [
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        # 最後一層 Dense 輸出: (batch_size, forecast_length)
        self.output_dense = Dense(forecast_length, activation='linear')

    def call(self, X, mask=None, past_preds=None, training=False):
        """
        X.shape:          (batch_size, seq_len, input_dim)
        mask.shape:       (batch_size, seq_len, 1) or None
        past_preds.shape: (batch_size, forecast_length)
                          or (batch_size, 1, forecast_length)
                          若要對應 seq_len, 可能需要先 tile
        """
        seq_len = tf.shape(X)[1]

        if past_preds is not None:
            # 先確保 past_preds 的 shape = (batch, 1, forecast_length)
            if len(past_preds.shape) == 2:
                # (batch, forecast_length) -> (batch, 1, forecast_length)
                past_preds = tf.expand_dims(past_preds, axis=1)

            # tile 到 seq_len -> (batch, seq_len, forecast_length)
            past_preds_tiled = tf.tile(past_preds, [1, seq_len, 1])
            # concat 在最後一維: (batch, seq_len, input_dim + forecast_length)
            x_in = tf.concat([X, past_preds_tiled], axis=-1)
        else:
            x_in = X  # (batch, seq_len, input_dim)

        # 投影到 embed_dim
        x_embed = self.input_projection(x_in)

        # 經過多層 Encoder
        for encoder in self.encoders:
            x_embed = encoder(x_embed, mask=mask, training=training)

        # 取最後一個 time step (batch_size, embed_dim)
        x_last = x_embed[:, -1, :]
        # 輸出 (batch_size, forecast_length)
        out = self.output_dense(x_last)
        return out


# -------------------------------------------------------------------
# 3) 實作: 讀取資料, 自動產生 group_id, 自訂訓練流程
# -------------------------------------------------------------------
def load_data_numpy(load_path="test_raw.npz"):
    """
    從 .npz 檔案讀取資料,
    假設其中有 X_data, y_data, mask_data
    形狀分別是:
    X_data:   (num_samples, seq_len, input_dim)
    y_data:   (num_samples, forecast_length)
    mask_data:(num_samples, seq_len)
    """
    data = np.load(load_path)
    return data["X_data"], data["y_data"], data["mask_data"]


def assign_group_ids(num_samples, group_size=8):
    """
    依照指定的 group_size,
    每 group_size 筆資料視為同一個 group
    如樣本1~8 -> group=0, 9~16->group=1, ...
    """
    group_ids = []
    group_id = 0
    for i in range(num_samples):
        group_ids.append(group_id)
        # 每滿 group_size 筆就換下一組
        if (i+1) % group_size == 0:
            group_id += 1
    return np.array(group_ids, dtype=np.int32)


# ========== 自訂訓練函式 (單筆 batch=1 為例) ==========
@tf.function
def train_step(model, optimizer, loss_fn, x, m, y, past_preds):
    """
    單筆訓練 step:
      x: (1, seq_len, input_dim)
      m: (1, seq_len) or (1, seq_len, 1)
      y: (1, forecast_length)
      past_preds: (1, forecast_length) or (1,1,forecast_length)
    """
    with tf.GradientTape() as tape:
        # forward pass
        preds = model(x, mask=m, past_preds=past_preds, training=True)

        loss_value = loss_fn(y, preds)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return preds, loss_value


def main():
    # 1) 讀取數據
    X_data, y_data, mask_data = load_data_numpy("dataset.npz")

    num_samples = X_data.shape[0]
    seq_len = X_data.shape[1]
    input_dim = X_data.shape[2]
    forecast_length = y_data.shape[1]

    # 2) 產生 group_id (僅示範, 與原邏輯相同)
    group_size = 8
    group_ids = assign_group_ids(num_samples, group_size=group_size)
    unique_groups = np.unique(group_ids)

    # 3) 建立模型
    model = RecurrentTransformerModel(
        input_dim=input_dim,
        forecast_length=forecast_length,
        num_layers=2,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        dropout_rate=0.1
    )

    # 4) 損失與優化器
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # 5) 開始訓練 (示範 epoch loop)
    EPOCHS = 5
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        steps = 0

        # (1) 每個 epoch 開始前，準備收集所有 preds 跟真值
        epoch_true_values = []
        epoch_predictions = []

        for g in unique_groups:
            idxs = np.where(group_ids == g)[0]
            past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

            for i in idxs:
                x_i = X_data[i][None, ...]
                y_i = y_data[i][None, ...]
                m_i = mask_data[i][None, ...]

                preds, loss_value = train_step(model, optimizer, loss_fn, x_i, m_i, y_i, past_preds)
                past_preds = preds

                # 收集 preds & y 的資料
                # preds.shape = (1, forecast_length)
                epoch_predictions.append(preds.numpy().flatten())
                epoch_true_values.append(y_i.flatten())


                epoch_loss += loss_value.numpy()
                steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0

        # (2) 計算 MAE、RMSE、R2
        if len(epoch_predictions) > 0:
            epoch_predictions_arr = np.vstack(epoch_predictions)
            epoch_true_values_arr = np.vstack(epoch_true_values)

            mae_val = mean_absolute_error(epoch_true_values_arr, epoch_predictions_arr)
            mse_val = mean_squared_error(epoch_true_values_arr, epoch_predictions_arr)
            rmse_val = np.sqrt(mse_val)
            r2_val = r2_score(epoch_true_values_arr, epoch_predictions_arr)

            print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.6f} | "
                  f"MAE: {mae_val:.6f}, RMSE: {rmse_val:.6f}, R2: {r2_val:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.6f}")

    print("訓練完成!")

    model.save(r"G:\GoldFX_Transformer\models\model_a", save_format="tf")
    print('模型已儲存為 "model_a"')


if __name__ == "__main__":
    main()


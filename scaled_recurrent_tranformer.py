import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm



# 1) TransformerEncoder (保持與之前相同)
class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
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
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            # (batch, seq_len, 1) -> (batch, seq_len)
            mask = tf.squeeze(mask, axis=-1)
            # -> (batch, 1, 1, seq_len)
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.expand_dims(mask, axis=1)
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

# 2) RecurrentTransformerModel
class RecurrentTransformerModel(keras.Model):
    def __init__(self, input_dim, forecast_length,
                 num_layers=2, embed_dim=64, num_heads=4,
                 ff_dim=128, dropout_rate=0.1):
        super().__init__()
        self.input_projection = Dense(embed_dim)
        self.encoders = [
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.final_dense = Dense(forecast_length)

    def call(self, X, mask=None, past_preds=None, training=False):
        seq_len = tf.shape(X)[1]
        if past_preds is not None:
            if len(past_preds.shape) == 2:
                past_preds = tf.expand_dims(past_preds, axis=1)
            past_preds_tiled = tf.tile(past_preds, [1, seq_len, 1])
            X = tf.concat([X, past_preds_tiled], axis=-1)

        x_embed = self.input_projection(X)
        for encoder in self.encoders:
            x_embed = encoder(x_embed, mask=mask, training=training)

        x_last = x_embed[:, -1, :]
        out = self.final_dense(x_last)
        return out

# 3) 自訂損失函數
def masked_mse_loss(y_true, y_pred, y_mask):
    """
    y_true, y_pred, y_mask 形狀: (batch_size, forecast_length)
    只對 y_mask=1 的位置計算 (y_true - y_pred)^2，其他位置忽略。
    """
    # element-wise (y_true - y_pred)^2
    sq_err = tf.square(y_true - y_pred)
    # masked
    masked_sq_err = sq_err * y_mask
    # 對每個樣本來說，sum起來之後，除以有效數量
    sum_err = tf.reduce_sum(masked_sq_err, axis=1)  # shape (batch,)
    valid_counts = tf.reduce_sum(y_mask, axis=1)    # (batch,)
    # 避免除以0
    valid_counts = tf.where(valid_counts == 0, 1., valid_counts)
    mse_per_sample = sum_err / valid_counts
    # 再對 batch 做平均
    return tf.reduce_mean(mse_per_sample)


# 3) 自訂 train step
@tf.function
def train_step(model, optimizer, x, m, y, y_mask, past_preds):
    with tf.GradientTape() as tape:
        preds = model(x, mask=m, past_preds=past_preds, training=True)
        # 自訂 masked loss
        loss = masked_mse_loss(y, preds, y_mask)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return preds, loss


def main():
    # === 讀取 scaled 的 train 資料 ===
    data_train = np.load("train_scaled.npz")
    X_train = data_train["X_data"]
    y_train = data_train["y_data"]
    m_train = data_train["mask_data"]
    y_mask_train = data_train["y_mask_data"]  # <--- 新增
    g_train = data_train["group_ids"]

    print("Train X shape:", X_train.shape)
    print("Train y shape:", y_train.shape)
    print("Train mask shape:", m_train.shape)
    print("Train group_ids shape:", g_train.shape)
    unique_train_groups = np.unique(g_train)
    print("Train group 數量:", len(unique_train_groups))

    # === 建立模型 & optimizer ===
    input_dim = X_train.shape[2]
    forecast_length = y_train.shape[1]
    model = RecurrentTransformerModel(
        input_dim=input_dim,
        forecast_length=forecast_length,
        num_layers=2,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        dropout_rate=0.1
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    EPOCHS = 10
    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):

        epoch_loss = 0.0
        steps = 0

        epoch_true_values = []
        epoch_predictions = []

        for g in unique_train_groups:
            idxs = np.where(g_train == g)[0]
            past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

            for i in idxs:
                x_i = X_train[i][None, ...]  # shape (1, seq_len, input_dim)
                y_i = y_train[i][None, ...]  # (1, forecast_length)
                m_i = m_train[i][None, ...]  # (1, seq_len, 1)
                y_mask_i = y_mask_train[i][None, ...]  # (1, forecast_length)

                preds, loss_val = train_step(
                    model, optimizer,
                    x_i, m_i, y_i, y_mask_i,
                    past_preds
                )
                past_preds = preds

                epoch_predictions.append(preds.numpy().flatten())
                epoch_true_values.append(y_i.flatten())

                epoch_loss += loss_val.numpy()
                steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0

        # (2) 計算 MAE, RMSE, R2
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

    model.save(r"G:\GoldFX_Transformer\models\model_b", save_format="tf")
    print('模型已儲存為 "model_b"')


if __name__ == "__main__":
    main()

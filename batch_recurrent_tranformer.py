import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


# -------------------- 1) TransformerEncoder --------------------
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


# -------------------- 2) RecurrentTransformerModel --------------------
class RecurrentTransformerModel(keras.Model):
    def __init__(self, input_dim, forecast_length,
                 num_layers=2, embed_dim=64, num_heads=4,
                 ff_dim=128, dropout_rate=0.1):
        super().__init__()
        # 注意：在 call() 中會把 past_preds concat 到 X
        self.input_projection = Dense(embed_dim)
        self.encoders = [
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.final_dense = Dense(forecast_length)

    def call(self, X, mask=None, past_preds=None, training=False):
        seq_len = tf.shape(X)[1]
        if past_preds is not None:
            # past_preds: (batch, forecast_length)
            # -> (batch, seq_len, forecast_length)
            if len(past_preds.shape)==2:
                past_preds = tf.expand_dims(past_preds, axis=1)
            past_preds_tiled = tf.tile(past_preds, [1, seq_len, 1])
            X = tf.concat([X, past_preds_tiled], axis=-1)

        x_embed = self.input_projection(X)
        for encoder in self.encoders:
            x_embed = encoder(x_embed, mask=mask, training=training)
        x_last = x_embed[:, -1, :]
        out = self.final_dense(x_last)
        return out


# -------------------- 3) 自訂損失函數 masked_mse_loss --------------------
def masked_mse_loss(y_true, y_pred, y_mask):
    sq_err = tf.square(y_true - y_pred)
    masked_sq_err = sq_err * y_mask
    sum_err = tf.reduce_sum(masked_sq_err, axis=1)
    valid_counts = tf.reduce_sum(y_mask, axis=1)
    valid_counts = tf.where(valid_counts==0, 1., valid_counts)
    mse_per_sample = sum_err / valid_counts
    return tf.reduce_mean(mse_per_sample)


# -------------------- 4) 單步訓練 train_step --------------------
@tf.function
def train_step(model, optimizer, x, m, y, y_mask, past_preds):
    with tf.GradientTape() as tape:
        preds = model(x, mask=m, past_preds=past_preds, training=True)
        loss = masked_mse_loss(y, preds, y_mask)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return preds, loss


# -------------------- 5) 讀取並組裝: 來自 data_preprocessing.py 的train_raw.npz -------------
def load_train_data(npz_path="train_raw.npz"):
    data = np.load(npz_path)
    X_data = data["X_data"]         # (num_samples, max_input_length, input_dim)
    y_data = data["y_data"]         # (num_samples, forecast_length)
    mask_data = data["mask_data"]   # (num_samples, max_input_length, 1)
    y_mask_data = data["y_mask_data"]  # (num_samples, forecast_length)
    group_ids = data["group_ids"]   # (num_samples,)

    return X_data, y_data, mask_data, y_mask_data, group_ids


# -------------------- 6) 主訓練邏輯: 以 group -> 逐序列 -> partial teacher forcing --------------------
def run_training_batch(
    X_data, y_data, mask_data, y_mask_data, group_ids,
    input_dim, forecast_length,
    epochs=10, lr=0.1
):
    unique_groups = np.unique(group_ids)
    print(f"共有 group 數量: {len(unique_groups)}")

    model = RecurrentTransformerModel(
        input_dim=input_dim,
        forecast_length=forecast_length,
        num_layers=4,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        dropout_rate=0.4
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        print(f"=== Epoch {epoch+1}/{epochs} ===")
        epoch_loss=0.0
        steps=0

        for g in unique_groups:
            idxs = np.where(group_ids==g)[0]
            # 這個 group 內有多筆序列, 順序= idxs
            # past_preds shape=(1, forecast_length), 針對單 batch
            past_preds = tf.zeros((1, forecast_length), dtype=tf.float32)

            for seq_idx, i in enumerate(idxs):
                # 取出該序列
                x_i = X_data[i][None, ...]   # (1, max_input_length, input_dim)
                y_i = y_data[i][None, ...]   # (1, forecast_length)
                m_i = mask_data[i][None, ...]# (1, max_input_length, 1)
                ym_i= y_mask_data[i][None,...]#(1, forecast_length)

                # 1) 取出新事實 close => e.g. 輸入最後一個close
                #    X_data[i, -1, 3] => shape=()
                new_fact_close = x_i[0,-1,3]  # tf tensor (scalar)

                # 2) 如果 seq_idx>0 => partial teacher forcing
                #    => 只替換 past_preds的第0位置 => new_fact_close
                if seq_idx>0:
                    # past_preds shape=(1, forecast_length)
                    # => scatter => shape=(1,)
                    # => indices=[[0,0]]
                    new_fact_close_1d = tf.reshape(new_fact_close, [1]) # shape=(1,)
                    idx_col = tf.constant([[0,0]], dtype=tf.int32)
                    past_preds_updated = tf.tensor_scatter_nd_update(
                        past_preds,
                        idx_col,
                        new_fact_close_1d
                    )
                else:
                    past_preds_updated = past_preds  # 第一個序列就不做 partial TF

                # 3) 執行 train_step
                preds_i, loss_i = train_step(
                    model, optimizer,
                    x_i, m_i, y_i, ym_i,
                    past_preds_updated
                )
                # 4) autoregressive => 下一個序列帶入 preds
                past_preds = preds_i

                epoch_loss += loss_i.numpy()
                steps += 1

        if steps>0:
            print(f"Epoch {epoch+1}, avg_loss={epoch_loss/steps:.6f}")
        else:
            print(f"Epoch {epoch+1}, no steps ???")

    return model


def main():
    X_data, y_data, mask_data, y_mask_data, group_ids = load_train_data("train_raw.npz")
    print("X_data:", X_data.shape, "y_data:", y_data.shape)
    unique_groups = np.unique(group_ids)
    print("Group數:", len(unique_groups))

    # 由 data_preprocessing 得到 shapes
    input_dim = X_data.shape[2]
    forecast_length = y_data.shape[1]

    model = run_training_batch(
        X_data, y_data, mask_data, y_mask_data, group_ids,
        input_dim=input_dim,
        forecast_length=forecast_length,
        epochs=10,      # 可自行調整
        lr=0.1
    )

    # (可選) 模型保存
    model.save("model_b_group_partialTF", save_format="tf")
    print("Done.")

if __name__=="__main__":
    main()

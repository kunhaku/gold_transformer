import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization
import numpy as np


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_heads = num_heads  # **儲存 num_heads**
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
            mask = tf.cast(mask, dtype=tf.float32)  # 確保 mask 是 float32
            mask = tf.squeeze(mask, axis=-1)  # 變成 (batch_size, seq_length)
            mask = tf.expand_dims(mask, axis=1)  # 變成 (batch_size, 1, seq_length)
            mask = tf.expand_dims(mask, axis=1)  # 變成 (batch_size, 1, 1, seq_length)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])  # **使用 self.num_heads**

        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(input_shape, forecast_length=10, num_layers=2, embed_dim=64, num_heads=4, ff_dim=128, dropout_rate=0.1):
    inputs = keras.Input(shape=input_shape)
    mask_inputs = keras.Input(shape=(input_shape[0], 1))  # Mask 輸入

    x = Dense(embed_dim)(inputs)

    for _ in range(num_layers):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)(x, mask=mask_inputs)

    # 修改 Dense 層的輸出維度，確保與 y_data 匹配
    outputs = Dense(forecast_length, activation="linear")(x[:, -1, :])

    model = keras.Model(inputs=[inputs, mask_inputs], outputs=outputs)
    return model



def load_data_numpy(load_path="dataset.npz"):
    data = np.load(load_path)
    return data["X_data"], data["y_data"], data["mask_data"]


def make_dataset(X, y, mask, batch_size=32):
    mask = np.expand_dims(mask, axis=-1)  # 確保 mask 形狀是 (batch_size, seq_length, 1)
    dataset = tf.data.Dataset.from_tensor_slices(((X, mask), y))  # **修正這裡**
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset




if __name__ == "__main__":
    # 讀取數據
    X_data, y_data, mask_data = load_data_numpy("dataset.npz")

    # 建立數據集
    train_dataset = make_dataset(X_data, y_data, mask_data)

    # 建立 Transformer 模型
    input_shape = (X_data.shape[1], X_data.shape[2])
    transformer_model = build_transformer_model(input_shape, forecast_length=y_data.shape[1])

    # 編譯模型
    transformer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    # 訓練模型
    transformer_model.fit(train_dataset, epochs=100)

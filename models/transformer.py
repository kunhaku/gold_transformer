from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)


class TransformerEncoder(Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, mask=None, training=False):
        attn_mask = None
        if mask is not None:
            attn_mask = tf.cast(mask, dtype=tf.float32)
            attn_mask = tf.squeeze(attn_mask, axis=-1)
            attn_mask = tf.expand_dims(attn_mask, axis=1)
            attn_mask = tf.expand_dims(attn_mask, axis=1)
            attn_mask = tf.tile(attn_mask, [1, self.num_heads, 1, 1])

        attn_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=attn_mask, training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class RecurrentTransformerModel(keras.Model):
    def __init__(
        self,
        input_dim: int,
        forecast_length: int,
        num_layers: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_projection = Dense(embed_dim)
        self.encoders = [
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
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
        return self.final_dense(x_last)

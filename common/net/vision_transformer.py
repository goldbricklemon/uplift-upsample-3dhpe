# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import einops.layers.keras


def drop_path(x, rate):
    keep_prob = 1. - rate
    batch_shape = tf.shape(x)[:1]
    rand_shape = tf.concat([batch_shape, tf.ones(tf.rank(x) - 1, dtype=tf.int32)], axis=0)
    rand = tf.random.uniform(shape=rand_shape, dtype=tf.float32)
    rand += keep_prob
    rand = tf.math.floor(rand)
    # This is different to the Stochastic Depth in tfa:
    # Instead of scaling the output by keep_prob during inference
    # The output is scaled by 1/keep_prob during training
    # This way, no rescaling is required in inference mode
    output = (x / keep_prob) * rand
    return output


class DropPath(kl.Layer):

    def __init__(self, rate=0., **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if self.rate == 0:
            return inputs
        if training:
            return drop_path(inputs, self.rate)
        else:
            return inputs


class MLP(kl.Layer):
    def __init__(self, out_features, hidden_features=None, activation=keras.activations.gelu, dropout=0., inner_dropout=0., **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.out_features = out_features
        self.hidden_features = out_features if hidden_features is None else hidden_features
        self.dropout = dropout
        self.inner_dropout = inner_dropout

        self.fc1 = kl.Dense(self.hidden_features)
        self.act = kl.Activation(activation)
        self.fc2 = kl.Dense(self.out_features)
        self.drop = kl.Dropout(dropout) if dropout > 0 else None
        self.inner_drop = kl.Dropout(inner_dropout) if inner_dropout > 0 else None

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        if self.inner_drop is not None:
            x = self.inner_drop(x, training=training)
        x = self.fc2(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        return x


class MHA(kl.Layer):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super(MHA, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        assert self.dim % self.num_heads == 0
        self.depth = self.dim // self.num_heads

        self.wq = kl.Dense(self.dim, use_bias=qkv_bias)
        self.wk = kl.Dense(self.dim, use_bias=qkv_bias)
        self.wv = kl.Dense(self.dim, use_bias=qkv_bias)

        self.projection = kl.Dense(self.dim)
        if self.attn_drop > 0:
            self.attn_dropout = kl.Dropout(rate=self.attn_drop)
        if self.proj_drop > 0:
            self.proj_dropout = kl.Dropout(rate=self.proj_drop)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, training=None, mask=None):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        if self.attn_drop > 0:
            attention_weights = self.attn_dropout(attention_weights, training=training)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        return output, attention_weights

    def call(self, v, k, q, training=None, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, dim)
        k = self.wk(k)  # (batch_size, seq_len, dim)
        v = self.wv(v)  # (batch_size, seq_len, dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, training=training, mask=mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dim))  # (batch_size, seq_len_q, dim)
        output = self.projection(concat_attention)  # (batch_size, seq_len_q, dim)

        if self.proj_drop > 0:
            output = self.proj_dropout(output, training=training)

        return output, attention_weights


class TransformerBlock(kl.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False,
                 attn_dropout=0., dropout=0.,
                 inner_dropout=0.,
                 drop_path=0., activation=keras.activations.gelu, norm_layer=kl.LayerNormalization,
                 return_attention=True, **kwargs):
        super(TransformerBlock, self).__init__(self, **kwargs)
        self.return_attention = return_attention
        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = MHA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_dropout, proj_drop=dropout)
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0. else None
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(out_features=dim, hidden_features=mlp_hidden_dim, activation=activation, dropout=dropout,
                       inner_dropout=inner_dropout)

    def call(self, x, pos_encoding=None, training=None, mask=None):
        """
        NOTE: PE needs to have the same size as x !!!!!
        """
        if pos_encoding is not None:
            assert x.shape[1] == pos_encoding.shape[1]
            x = x + pos_encoding
        y = self.norm1(x)
        y, attn = self.attn(y, y, y, training=training,  mask=mask)
        if self.drop_path_layer is not None:
            y = self.drop_path_layer(y)
        x = x + y
        z = self.mlp(self.norm2(x))
        if self.drop_path_layer is not None:
            z = self.drop_path_layer(z)
        x = x + z
        if self.return_attention:
            return x, attn
        else:
            return x

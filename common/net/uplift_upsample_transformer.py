# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import einops.layers.keras

from common.net import vision_transformer as vit


class LearnablePELayer(keras.layers.Layer):

    def __init__(self, shape, **kwargs):
        super(LearnablePELayer, self).__init__(**kwargs)
        channel_str = " ".join([f"d{i}" for i in range(len(shape))])
        self.repeat_str = f"{channel_str} -> b {channel_str}"
        self.pe = self.add_weight(name=f"{self.name}/positional_encoding_weights",
                                  shape=shape,
                                  trainable=True,
                                  initializer=keras.initializers.TruncatedNormal(stddev=.02))

    def call(self, inputs):
        b = inputs.shape[0]
        batched_pe = einops.repeat(self.pe, self.repeat_str, b=b)
        return batched_pe


class LearnableMaskedTokenLayer(keras.layers.Layer):

    def __init__(self, dim, **kwargs):
        super(LearnableMaskedTokenLayer, self).__init__(**kwargs)
        self.learnable_token = self.add_weight(name=f"{self.name}/learnable_masked_token",
                                               shape=(dim,),
                                               trainable=True,
                                               initializer=keras.initializers.TruncatedNormal(stddev=.02))

    def call(self, inputs):
        b, n = inputs.shape[0], inputs.shape[1]
        repeated_token = einops.repeat(self.learnable_token, "c -> b n c", b=b, n=n)
        return repeated_token


class StridedMLP(kl.Layer):
    def __init__(self, out_features, hidden_features=None, activation=keras.activations.gelu,
                 dropout=0., inner_dropout=0.,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 **kwargs):
        super(StridedMLP, self).__init__(**kwargs)
        self.out_features = out_features
        self.hidden_features = out_features if hidden_features is None else hidden_features
        self.dropout = dropout
        self.inner_dropout = inner_dropout
        self.stride = stride

        self.fc1 = kl.Conv1D(filters=hidden_features, kernel_size=1)
        self.act = kl.Activation(activation)
        if padding is None:
            pad = kernel_size // 2
        elif type(padding) is int:
            pad = padding
        else:
            pad = (padding[0], padding[1])
        self.zero_pad = kl.ZeroPadding1D(padding=pad)
        self.strided_conv = kl.Conv1D(filters=out_features, kernel_size=kernel_size, strides=stride,
                                      padding="valid")
        self.drop = kl.Dropout(dropout) if dropout > 0 else None
        self.inner_drop = kl.Dropout(inner_dropout) if inner_dropout > 0 else None

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        if self.inner_drop is not None:
            x = self.inner_drop(x, training=training)
        x = self.zero_pad(x)
        x = self.strided_conv(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        return x


class StridedTransformerBlock(kl.Layer):

    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False,
                 attn_dropout=0., dropout=0.,
                 inner_dropout=0.,
                 drop_path=0., activation=keras.activations.gelu, norm_layer=kl.LayerNormalization,
                 kernel_size=3,
                 stride=3,
                 padding=None,
                 return_attention=True, **kwargs):
        super(StridedTransformerBlock, self).__init__(self, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_attention = return_attention
        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = vit.MHA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_dropout, proj_drop=dropout)
        self.drop_path_layer = vit.DropPath(drop_path) if drop_path > 0. else None
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = StridedMLP(out_features=dim, hidden_features=mlp_hidden_dim, activation=activation, dropout=dropout,
                              inner_dropout=inner_dropout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.max_pool = None
        if stride > 1:
            # NOTE: The strange pool size of 1 is taken directly from the authors original code!
            # They confirmed this in a github issue.
            self.max_pool = kl.MaxPool1D(pool_size=1, strides=stride)

    def call(self, x, pos_encoding=None, training=None, mask=None):
        """
        NOTE: PE needs to have the same size as x !!!!!
        """
        if pos_encoding is not None:
            assert x.shape[1] == pos_encoding.shape[1]
            x = x + pos_encoding
        y = self.norm1(x)
        y, attn = self.attn(y, y, y, training=training, mask=mask)
        if self.drop_path_layer is not None:
            y = self.drop_path_layer(y)
        x = x + y
        z = self.mlp(self.norm2(x))
        if self.drop_path_layer is not None:
            z = self.drop_path_layer(z)

        if self.max_pool is not None:
            if self.padding is None:
                pad = (self.kernel_size // 2, self.kernel_size // 2)
            elif type(self.padding) is int:
                pad = (self.padding, self.padding)
            else:
                pad = (self.padding[0], self.padding[1])

            identity = x
            if pad[0] == 0:
                identity = identity[:, 1:]

            if pad[1] == 0:
                identity = identity[:, :-1]
            identity = self.max_pool(identity)
        else:
            identity = x

        x = identity + z
        if self.return_attention:
            return x, attn
        else:
            return x


class UpliftUpsampleTransformer(keras.Model):

    def __init__(self, full_output=True, num_frames=9, num_keypoints=17,
                 spatial_d_model=16, temporal_d_model=256,
                 spatial_depth=3, temporal_depth=3,
                 strides=[3, 3, 3],
                 paddings=None,
                 num_heads=8, mlp_ratio=2., qkv_bias=True,
                 attn_drop_rate=0.1, drop_rate=0.1, drop_path_rate=0.0, norm_layer=kl.LayerNormalization,
                 output_bn=False,
                 has_strided_input=False,
                 first_strided_token_attention_layer=0,
                 token_mask_rate=0., learnable_masked_token=False,
                 return_attention=False,
                 **kwargs):

        super(UpliftUpsampleTransformer, self).__init__(**kwargs)

        out_dim = num_keypoints * 3

        self.full_output = full_output
        self.num_frames = num_frames
        self.spatial_d_model = spatial_d_model
        self.temporal_d_model = temporal_d_model
        self.spatial_depth = spatial_depth
        self.temporal_depth = temporal_depth
        self.strides = strides
        self.has_strided_input = has_strided_input
        self.first_strided_token_attention_layer = first_strided_token_attention_layer
        self.token_mask_rate = token_mask_rate
        self.learnable_masked_token = learnable_masked_token
        self.return_attention = return_attention

        # Keypoint embedding and PE
        if self.spatial_depth > 0:
            self.keypoint_embedding = kl.Dense(spatial_d_model, name="keypoint_embedding")
        # Note that "token_dropout" might be misleading
        # It does not drop complete tokens, but performs standard dropout (independent across all axes)
        self.token_dropout = kl.Dropout(rate=drop_rate, name="token_dropout")

        if spatial_depth > 0:
            self.spatial_pos_encoding = LearnablePELayer(shape=(num_keypoints, self.spatial_d_model),
                                                         name="spatial_pe")
        self.temporal_pos_encoding = LearnablePELayer(shape=(self.num_frames, self.temporal_d_model),
                                                      name="temporal_pe")
        self.strided_temporal_pos_encodings = []
        if len(self.strides) > 0:
            seq_len = self.num_frames
            for i, s in enumerate(self.strides):
                p = [1, 1] if paddings is None else paddings[i]
                pe_shape = (seq_len, self.temporal_d_model)
                self.strided_temporal_pos_encodings.append(
                    LearnablePELayer(shape=pe_shape, name=f"strided_temporal_pe_{i + 1}"))
                seq_len = math.ceil((seq_len + p[0] + p[1] - 2) / s)

        # Token masking. This is the actual dropout on token level
        if token_mask_rate > 0 and learnable_masked_token is True:
            self.learnable_masked_token_layer = LearnableMaskedTokenLayer(dim=self.temporal_d_model)

        # Masked input token
        if self.has_strided_input is True:
            self.learnable_strided_input_token_layer = LearnableMaskedTokenLayer(dim=self.temporal_d_model,
                                                                                 name=f"strided_input_token_layer")

        # Spatial blocks
        if self.spatial_depth > 0:
            dpr = drop_path_rate[0] if type(drop_path_rate) is list else drop_path_rate
            path_drop_rates = np.linspace(0, dpr, self.spatial_depth)
            self.spatial_blocks = [
                vit.TransformerBlock(dim=self.spatial_d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, attn_dropout=attn_drop_rate,
                                     dropout=drop_rate, drop_path=path_drop_rates[i], norm_layer=norm_layer,
                                     return_attention=True, name=f"spatial_block_{i + 1}")
                for i in range(self.spatial_depth)]

            self.spatial_norm = norm_layer(epsilon=1e-6, name="spatial_norm")
        self.spatial_to_temporal_mapping = kl.Dense(self.temporal_d_model, name="spatial_to_temporal_fc")

        # Full sequence temporal blocks
        if self.temporal_depth > 0:
            dpr = drop_path_rate[1] if type(drop_path_rate) is list else drop_path_rate
            path_drop_rates = np.linspace(0, dpr, self.temporal_d_model)
            self.temporal_blocks = [
                vit.TransformerBlock(dim=self.temporal_d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, attn_dropout=attn_drop_rate,
                                     dropout=drop_rate, inner_dropout=drop_rate,
                                     drop_path=path_drop_rates[i],
                                     activation=keras.activations.relu,
                                     norm_layer=norm_layer,
                                     return_attention=True, name=f"temporal_block_{i + 1}")
                for i in range(self.temporal_depth)]

        # Strided temporal blocks
        if len(self.strides) > 0:
            pad_values = paddings
            if paddings is None:
                pad_values = [None] * len(strides)

            dpr = drop_path_rate[2] if type(drop_path_rate) is list else drop_path_rate
            path_drop_rates = np.linspace(0, dpr, len(strides))
            self.strided_temporal_blocks = [
                StridedTransformerBlock(dim=temporal_d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, attn_dropout=attn_drop_rate,
                                        dropout=drop_rate, inner_dropout=drop_rate,
                                        stride=s, kernel_size=3,
                                        padding=pad_values[i],
                                        drop_path=path_drop_rates[i],
                                        activation=keras.activations.relu,
                                        norm_layer=norm_layer,
                                        return_attention=True, name=f"strided_temporal_block_{i + 1}")
                for i, s in enumerate(strides)]

        self.head1 = None
        if self.full_output is True and self.temporal_depth > 0:
            self.head1 = []
            if output_bn:
                self.head1.append(kl.BatchNormalization(momentum=0.1, epsilon=1e-5, axis=-1, name="temporal_norm"))
            self.head1.append(kl.Dense(units=3 * num_keypoints, name="temporal_fc"))

        self.head2 = []
        if output_bn:
            self.head2.append(kl.BatchNormalization(momentum=0.1, epsilon=1e-5, axis=-1, name="strided_temporal_norm"))
        self.head2.append(kl.Dense(units=out_dim, name="strided_temporal_fc"))

    def random_token_masking(self, x, masked_token_value):
        # Input has shape B, N, C
        b, n, c = x.shape
        # Ensure that central frame is never masked
        mid_index = self.num_frames // 2
        center_mask = tf.range(start=0, limit=self.num_frames, dtype=tf.int32)
        center_mask = tf.not_equal(center_mask, mid_index)
        center_mask = einops.repeat(center_mask, "n -> b n", b=b)

        # Draw random token masking
        # If mask is 1, token will be masked
        # Mask has shape B,N
        token_mask = tf.random.uniform(shape=(b, n), maxval=1., dtype=tf.float32)
        token_mask = token_mask < self.token_mask_rate

        # Merge token mask and center mask
        token_mask = tf.logical_and(center_mask, token_mask)

        # Mask: B,N,C
        token_mask = tf.cast(tf.expand_dims(token_mask, axis=-1), dtype=tf.float32)
        inv_token_mask = 1. - token_mask

        # Mask token
        output = (x * inv_token_mask) + (masked_token_value * token_mask)
        return output

    def spatial_transformation(self, x, training=None, stride_mask=None):
        b, n, p, c = x.shape
        if self.spatial_depth == 0:
            x = einops.rearrange(x, "b n p c -> b n (p c)")
        else:
            # Fuse batch size and frames for frame-independent processing
            x = einops.rearrange(x, "b n p c -> (b n) p c")
            # ToDo: Handle stride mask with gather/scatter for efficiency during training
            x = self.keypoint_embedding(x)
            batched_sp_pe = self.spatial_pos_encoding(x)
            x += batched_sp_pe
            x = self.token_dropout(x, training=training)

            pe = None
            for block in self.spatial_blocks:
                x, att = block(x, pos_encoding=pe, training=training)
            x = self.spatial_norm(x)
            x = einops.rearrange(x, "(b n) p c -> b n (p c)", n=n)

        x = self.spatial_to_temporal_mapping(x)
        return x

    def temporal_transformation(self, x, training=None, stride_mask=None):
        if training is True and self.token_mask_rate > 0:
            masked_token_value = 0. if self.learnable_masked_token is False else self.learnable_masked_token_layer(x)
            x = self.random_token_masking(x, masked_token_value=masked_token_value)

        batched_temp_pe = self.temporal_pos_encoding(x)

        if self.has_strided_input is True:
            # (B, N, C)
            strided_input_token = self.learnable_strided_input_token_layer(batched_temp_pe)
            # (B, N)
            # Stride mask is 1 on valid (i.e. non-masked) indices !!!
            stride_mask = tf.cast(stride_mask, dtype=tf.float32)
            inv_stride_mask = 1. - stride_mask
            # Masked input token (B, N, C)
            x = (stride_mask[..., tf.newaxis] * x) + (inv_stride_mask[..., tf.newaxis] * strided_input_token)

        x += batched_temp_pe

        pe = None
        att_list = []
        if self.temporal_depth > 0:
            for i, block in enumerate(self.temporal_blocks):
                if self.has_strided_input and i < self.first_strided_token_attention_layer:
                    # Use inverted stride_mask to disable attention on the strided input tokens
                    # Must be broadcastable to (B, HEADS, QUERIES, KEYS)
                    attn_mask = inv_stride_mask[:, tf.newaxis, tf.newaxis, :]
                else:
                    attn_mask = None
                x, att = block(x, pos_encoding=pe, training=training, mask=attn_mask)
                att_list.append(att)
        # x is (B, N, C)
        return x, att_list

    def strided_temporal_transformation(self, x, training=None, stride_mask=None):
        b, n, c = x.shape

        for i, block in enumerate(self.strided_temporal_blocks):
            if self.temporal_depth == 0 and self.has_strided_input and i < self.first_strided_token_attention_layer:
                # Use inverted stride_mask to disable attention on the strided input tokens
                # Must be broadcastable to (B, HEADS, QUERIES, KEYS)
                stride_mask = tf.cast(stride_mask, dtype=tf.float32)
                inv_stride_mask = 1. - stride_mask
                attn_mask = inv_stride_mask[:, tf.newaxis, tf.newaxis, :]
                print(
                    "NOTE: Without temporal transformer blocks, deferred upsampling token attention will be used in strided transformer.")
            else:
                attn_mask = None
            pe = self.strided_temporal_pos_encodings[i](x)
            x, att = block(x, pos_encoding=pe, training=training, mask=attn_mask)
        # x is (B, N, C)
        return x

    def call(self, inputs, training=None, mask=None):
        if self.has_strided_input:
            x, stride_mask = inputs[0], inputs[1]
        else:
            x = inputs
            stride_mask = None
        b, n, p, _ = x.shape
        x = self.spatial_transformation(x, training=training, stride_mask=stride_mask)
        # Full sequence temporal transformer
        x, att_list = self.temporal_transformation(x, training=training, stride_mask=stride_mask)
        # Prediction for full sequence
        full_output = None
        if self.full_output is True and self.temporal_depth > 0:
            full_output = x
            for layer in self.head1:
                full_output = layer(full_output, training=training)
            full_output = einops.rearrange(full_output, "b n (p c) -> b n p c", p=p, c=3)

        # Strided transformer
        if len(self.strides) > 0:
            x = self.strided_temporal_transformation(x, training=training, stride_mask=stride_mask)
            # Prediction for central frame
            central_output = x
        else:
            central_output = x[:, self.num_frames // 2, :]
            central_output = central_output[:, tf.newaxis, :]
        for layer in self.head2:
            central_output = layer(central_output, training=training)
        central_output = einops.rearrange(central_output, "b n (p c) -> (b n) p c", n=1, p=p, c=3)

        if self.return_attention:
            return full_output, central_output, att_list
        else:
            return full_output, central_output

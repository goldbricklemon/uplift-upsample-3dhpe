# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

from common.net.uplift_upsample_transformer_config import UpliftUpsampleConfig
from common.net.uplift_upsample_transformer import UpliftUpsampleTransformer


def build_uplift_upsample_transformer(config: UpliftUpsampleConfig, **kwargs):
    input_shape = (config.SEQUENCE_LENGTH, config.NUM_KEYPOINTS, 2)
    has_strided_input = config.MASK_STRIDE is not None
    if has_strided_input:
        if type(config.MASK_STRIDE) is int and config.MASK_STRIDE == 1:
            has_strided_input = False
        if type(config.MASK_STRIDE) is list and config.MASK_STRIDE[0] == 1:
            has_strided_input = False

    model = UpliftUpsampleTransformer(full_output=not config.USE_REFINE,
                                      num_frames=config.SEQUENCE_LENGTH,
                                      num_keypoints=config.NUM_KEYPOINTS,
                                      spatial_d_model=config.SPATIAL_EMBED_DIM,
                                      temporal_d_model=config.TEMPORAL_EMBED_DIM,
                                      spatial_depth=config.SPATIAL_TRANSFORMER_BLOCKS,
                                      temporal_depth=config.TEMPORAL_TRANSFORMER_BLOCKS,
                                      strides=config.STRIDES,
                                      paddings=config.PADDINGS,
                                      num_heads=config.NUM_HEADS,
                                      mlp_ratio=config.MLP_RATIO,
                                      qkv_bias=config.QKV_BIAS,
                                      attn_drop_rate=config.ATTENTION_DROP_RATE,
                                      drop_rate=config.DROP_RATE,
                                      drop_path_rate=config.DROP_PATH_RATE,
                                      output_bn=config.OUTPUT_BN,
                                      has_strided_input=has_strided_input,
                                      first_strided_token_attention_layer=config.FIRST_STRIDED_TOKEN_ATTENTION_LAYER,
                                      token_mask_rate=config.TOKEN_MASK_RATE,
                                      learnable_masked_token=config.LEARNABLE_MASKED_TOKEN,
                                      **kwargs)
    batched_input_shape = (config.BATCH_SIZE,) + input_shape
    if model.has_strided_input is True:
        batched_mask_shape = (config.BATCH_SIZE, config.SEQUENCE_LENGTH)
        model.build([batched_input_shape, batched_mask_shape])
    else:
        model.build(batched_input_shape)
    return model

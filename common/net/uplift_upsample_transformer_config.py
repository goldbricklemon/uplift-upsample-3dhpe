# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

from common.utils.config import Config


class UpliftUpsampleConfig(Config):

    # Execution
    GPU_ID = 0
    BATCH_SIZE = 256

    ARCH = "UpliftUpsampleTransformer"

    SHUFFLE_SEED = 0

    SPATIAL_EMBED_DIM = 32
    TEMPORAL_EMBED_DIM = 348

    MLP_RATIO = 2
    NUM_HEADS = 8
    SPATIAL_TRANSFORMER_BLOCKS = 4
    TEMPORAL_TRANSFORMER_BLOCKS = 4
    STRIDES = [3, 3, 3]
    PADDINGS = None # equals [[1 1], [1, 1], [1, 1]]
    QKV_BIAS = True
    DROP_PATH_RATE = [0.1, 0.1, 0.0]
    DROP_RATE = 0.0
    ATTENTION_DROP_RATE = 0.0
    OUTPUT_BN = False

    # Refine module
    USE_REFINE = False
    REFINE_FC_SIZE = 1024
    REFINE_DROP_RATE = 0.5

    # Token Masking
    TOKEN_MASK_RATE = 0.
    LEARNABLE_MASKED_TOKEN = False

    # Objective
    NUM_KEYPOINTS = 17
    SEQUENCE_LENGTH = 27
    PADDING_TYPE = "copy"
    SEQUENCE_STRIDE = 1
    TEST_STRIDED_EVAL = True

    MASK_STRIDE = None
    STRIDE_MASK_RAND_SHIFT = False
    FIRST_STRIDED_TOKEN_ATTENTION_LAYER = 0

    LOSS_WEIGHT_SEQUENCE = 1.0
    LOSS_WEIGHT_CENTER = 1.0

    # Data handling and augmentation
    ROOT_KEYTPOINT = 6

    AUGM_FLIP_KEYPOINT_ORDER = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 16, 15, 14, 13, 12, 11]
    AUGM_FLIP_PROB = 0.5
    IN_BATCH_AUGMENT = False

    # Training
    EPOCHS = 120
    STEPS_PER_EPOCH = 6000

    DATASET_TRAIN_3D_SUBSAMPLE_STEP = 1
    DATASET_VAL_3D_SUBSAMPLE_STEP = 4
    DATASET_TEST_3D_SUBSAMPLE_STEP = 1

    # Validation
    VALIDATION_INTERVAL = 1
    # Number of validation examples independent of the batch size.
    # Set to -1 to use all validation examples
    VALIDATION_EXAMPLES = -1
    # Test time flip
    EVAL_FLIP = True
    # Diable learned upsampling
    EVAL_DISABLE_LEARNED_UPSAMPLING = False

    # Optimizer and Schedule
    OPTIMIZER = "Adam"
    OPTIMIZER_PARAMS = {"amsgrad": True,
                        "epsilon": 1e-08}

    SCHEDULE = "ExponentialDecayWithSteps"
    SCHEDULE_PARAMS = {
        "initial_learning_rate": 1e-3,
        "decay_steps": 12000,
        "decay_rate": 0.95,
        "large_decay_steps": 60000,
        "large_decay_rate":  0.5,
        }
    WEIGHT_DECAY = None

    EMA_ENABLED = False
    EMA_DECAY = None

    # Checkpoints
    CHECKPOINT_INTERVAL = 10
    BEST_CHECKPOINT_METRIC = "AW-MPJPE"
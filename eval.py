# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import os
import sys

import numpy as np
import time
import datetime
import tensorflow as tf

from common.utils import path_utils
from common.utils import time_formatting
from common.net.uplift_upsample_transformer_config import UpliftUpsampleConfig
from common.dataset.uplifiting_dataset import load_dataset_and_2d_poses, filter_and_subsample_dataset, \
    H36mSequenceGenerator
from common.dataset import h36m_splits
from common.net.uplift_upsample_transformer_constructor import build_uplift_upsample_transformer
from common.utils import weight_io
from common.dataset.action_wise_eval import compute_and_log_metrics, interpolate_between_keyframes


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_eval(config: UpliftUpsampleConfig, dataset_name, dataset_path, dataset2d_path, test_subset,
             weights_path=None, model=None, action_wise=True):
    """
    Run H3.6m evaluation with the given model.
    :param config: Model config
    :param dataset: dataset name
    :param dataset_path: 3D dataset .npz
    :param dataset2d_path: 2D dataset .npz
    :param test_subset: Dataset split to evaluate on
    :param weights_path: Path to weight file to load (optional).
    :param model: Model to evaluate.
    :param action_wise: Perform action-wise evaluation (True) or frame-wise evaluation (False)
    :return:
    """

    assert not (weights_path is None and model is None)

    # Build model, optimizer, checkpoint
    if model is None:
        tf_device = "/gpu:0"
        with tf.device(tf_device):
            model = build_uplift_upsample_transformer(config=config)
            if weights_path is not None:
                log(f"Loading weights from {weights_path}")
                weight_io.load_weights_with_callback(model, filepath=weights_path, skip_mismatch=False, verbose=True)

    elif weights_path is not None:
        log(f"Using provided model. Ignoring the given weights path: {weights_path}")

    @tf.function
    def test_step(keypoints2d, stride_masks):
        model_input = keypoints2d
        if model.has_strided_input:
            masked_keypoints2d = keypoints2d * tf.cast(stride_masks[:, :, tf.newaxis, tf.newaxis], dtype=tf.float32)
            model_input = [masked_keypoints2d, stride_masks]

        pred_keypoints_3d, pred_keypoints_3d_central = model(model_input, training=False)
        return pred_keypoints_3d, pred_keypoints_3d_central

    # Build h36m dataset
    if dataset_name == "h36m":
        selected_subjects = h36m_splits.subjects_by_split[test_subset]
    else:
        raise Exception("Invalid Dataset")
    dataset_3d, poses_2d_dataset = load_dataset_and_2d_poses(dataset_path=dataset_path,
                                                             poses_2d_path=dataset2d_path,
                                                             dataset_name=dataset_name,
                                                             verbose=True)

    # The dataset is subsampled to every Nth frame
    subsample = config.DATASET_TEST_3D_SUBSAMPLE_STEP

    action = "*"
    camera_params, poses_3d, poses_2d, _, sequence_subjects, sequence_actions, sequence_frame_rates = filter_and_subsample_dataset(
        dataset=dataset_3d,
        poses_2d=poses_2d_dataset,
        subjects=selected_subjects,
        action_filter=action,
        downsample=1,
        image_base_path=dataset_path,
        verbose=True)

    target_frame_rate = 50
    generator = H36mSequenceGenerator(poses_3d, poses_2d,
                                      camera_params=camera_params,
                                      subjects=sequence_subjects,
                                      actions=sequence_actions,
                                      frame_rates=sequence_frame_rates,
                                      split=test_subset,
                                      seq_len=config.SEQUENCE_LENGTH,
                                      target_frame_rate=target_frame_rate,
                                      subsample=subsample,
                                      stride=config.SEQUENCE_STRIDE,
                                      padding_type=config.PADDING_TYPE,
                                      mask_stride=config.MASK_STRIDE,
                                      stride_mask_align_global=True,
                                      rand_shift_stride_mask=False,
                                      flip_augment=False,
                                      shuffle=False)

    log(f"Sequences: {len(generator)}")

    output_sig = (tf.TensorSpec(shape=(config.SEQUENCE_LENGTH, config.NUM_KEYPOINTS, 3), dtype=tf.float32),
                  tf.TensorSpec(shape=(config.SEQUENCE_LENGTH, config.NUM_KEYPOINTS, 2), dtype=tf.float32),
                  tf.TensorSpec(shape=(config.SEQUENCE_LENGTH,), dtype=tf.float32),
                  tf.TensorSpec(shape=(11,), dtype=tf.float32),
                  tf.TensorSpec(shape=(), dtype=tf.int32),
                  tf.TensorSpec(shape=(), dtype=tf.int32),
                  tf.TensorSpec(shape=(), dtype=tf.int32),
                  tf.TensorSpec(shape=(config.SEQUENCE_LENGTH,), dtype=tf.bool),
                  )
    dataset = tf.data.Dataset.from_generator(generator.next_epoch_iterator, output_signature=output_sig)

    num_test_examples = len(generator)

    test_batches = np.ceil(num_test_examples / config.BATCH_SIZE)
    # Repeat once such that the last fractional batch can be extracted
    dataset = dataset.repeat(2)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.take(test_batches)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = dataset

    # Test loop
    log(f"Running evaluation on '{test_subset}' with {num_test_examples} examples")
    start = time.time()
    test_gt_keypoints3d = list()
    test_pred_keypoints3d = list()
    test_gt_subjects = list()
    test_gt_actions = list()
    test_gt_indices = list()
    examples = 0
    mid_index = config.SEQUENCE_LENGTH // 2
    for b_i, (
            test_sequences_3d, test_sequences_2d, test_sequences_mask, test_sequence_camera_params,
            test_sequence_subjects,
            test_sequence_actions, test_index, test_stride_masks) in enumerate(
        test_dataset):
        pred_sequence_keypoints3d, pred_keypoints3d = test_step(keypoints2d=test_sequences_2d,
                                                                stride_masks=test_stride_masks)
        if config.EVAL_FLIP is True:
            flipped_sequences_2d = test_sequences_2d
            flipped_sequences_2d = tf.concat([flipped_sequences_2d[:, :, :, :1] * -1.,
                                              flipped_sequences_2d[:, :, :, 1:]], axis=-1)
            flipped_sequences_2d = tf.gather(flipped_sequences_2d, indices=config.AUGM_FLIP_KEYPOINT_ORDER, axis=2)

            flipped_pred_sequence_keypoints_3d, flipped_pred_keypoints_3d = test_step(keypoints2d=flipped_sequences_2d,
                                                                                      stride_masks=test_stride_masks)

            flipped_pred_keypoints_3d = tf.concat([flipped_pred_keypoints_3d[:, :, :1] * -1.,
                                                   flipped_pred_keypoints_3d[:, :, 1:]], axis=-1)
            flipped_pred_keypoints_3d = tf.gather(flipped_pred_keypoints_3d, indices=config.AUGM_FLIP_KEYPOINT_ORDER,
                                                  axis=1)

            pred_keypoints3d += flipped_pred_keypoints_3d
            pred_keypoints3d /= 2.

            if model.full_output and config.TEMPORAL_TRANSFORMER_BLOCKS > 0:
                flipped_pred_sequence_keypoints_3d = tf.concat([flipped_pred_sequence_keypoints_3d[:, :, :, :1] * -1.,
                                                                flipped_pred_sequence_keypoints_3d[:, :, :, 1:]],
                                                               axis=-1)
                flipped_pred_sequence_keypoints_3d = tf.gather(flipped_pred_sequence_keypoints_3d,
                                                               indices=config.AUGM_FLIP_KEYPOINT_ORDER,
                                                               axis=2)

                pred_sequence_keypoints3d += flipped_pred_sequence_keypoints_3d
                pred_sequence_keypoints3d /= 2.

        # Only collect as many examples as needed
        examples_to_include = min(config.BATCH_SIZE, num_test_examples - examples)
        # Perform root-shift right before metric calculation
        test_sequences_3d = test_sequences_3d - test_sequences_3d[:, :,
                                                config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
        test_central_keypoints_3d = test_sequences_3d[:, mid_index]
        test_gt_keypoints3d.extend(test_central_keypoints_3d[:examples_to_include].numpy())
        test_pred_keypoints3d.extend(pred_keypoints3d[:examples_to_include].numpy())
        test_gt_subjects.extend(test_sequence_subjects[:examples_to_include].numpy())
        test_gt_actions.extend(test_sequence_actions[:examples_to_include].numpy())
        test_gt_indices.extend(test_index[:examples_to_include].numpy())
        examples += examples_to_include

    test_gt_keypoints3d = np.stack(test_gt_keypoints3d, axis=0).astype(np.float64)
    # Add dummy valid flag
    test_gt_keypoints3d = np.concatenate([test_gt_keypoints3d, np.ones(test_gt_keypoints3d.shape[:-1] + (1,))],
                                         axis=-1)
    test_pred_keypoints3d = np.stack(test_pred_keypoints3d, axis=0).astype(np.float64)

    test_gt_subjects = np.stack(test_gt_subjects, axis=0)
    test_gt_actions = np.stack(test_gt_actions, axis=0)
    test_gt_indices = np.stack(test_gt_indices, axis=0)
    assert b_i == (test_batches - 1)

    bkup_test_pred_keypoints3d = test_pred_keypoints3d
    test_pred_keypoints3d = np.copy(bkup_test_pred_keypoints3d)

    if config.SEQUENCE_STRIDE > 1 and config.TEST_STRIDED_EVAL is True:
        log(f"Performing strided eval: Interpolating between keyframes")
        strides = np.tile([config.SEQUENCE_STRIDE], reps=(test_gt_indices.shape[0]))
        if config.EVAL_DISABLE_LEARNED_UPSAMPLING and config.MASK_STRIDE is not None:
            strides[:] = config.MASK_STRIDE

        interp_pred_keypoints3d, _ = interpolate_between_keyframes(pred3d=test_pred_keypoints3d,
                                                                   frame_indices=test_gt_indices,
                                                                   keyframe_stride=strides)

        full_pred_keypoints3d = test_pred_keypoints3d
        test_pred_keypoints3d = interp_pred_keypoints3d
    else:
        full_pred_keypoints3d = test_pred_keypoints3d

    log("")
    log("### Evaluation on ALL FRAMES ####")
    log("")

    if dataset_name == "h36m":
        # Run H36m 3D evaluation
        compute_and_log_metrics(pred3d=test_pred_keypoints3d, gt3d=test_gt_keypoints3d,
                                actions=test_gt_actions, root_index=config.ROOT_KEYTPOINT,
                                action_wise=action_wise)
    else:
        raise Exception("Invalid Dataset")

    if (config.SEQUENCE_STRIDE > 1 or (
            config.MASK_STRIDE is not None and config.MASK_STRIDE > 1)) and config.TEST_STRIDED_EVAL is True:
        log("")
        log("### Evaluation on KEYFRAMES ####")
        log("")

        input_stride = config.SEQUENCE_STRIDE if config.MASK_STRIDE is None else config.MASK_STRIDE
        input_keyframes = np.equal(np.mod(test_gt_indices, input_stride), 0)

        if dataset_name == "h36m":
            compute_and_log_metrics(pred3d=full_pred_keypoints3d[input_keyframes],
                                    gt3d=test_gt_keypoints3d[input_keyframes],
                                    actions=test_gt_actions[input_keyframes], root_index=config.ROOT_KEYTPOINT,
                                    action_wise=action_wise)
        else:
            raise Exception("Invalid Dataset")

    duration = time.time() - start
    duration_string = time_formatting.format_time(duration)
    log(f"Finished evaluation in {duration_string}")


def run_eval_multi_mask_stride(config: UpliftUpsampleConfig, *args, **kwargs):
    # Run evaluation for each mask stride value
    config = config.copy()
    mask_stride_values = config.MASK_STRIDE
    if type(mask_stride_values) is not list:
        mask_stride_values = [mask_stride_values]
    for msv in mask_stride_values:
        config.MASK_STRIDE = msv
        if len(mask_stride_values) > 1:
            log(f"### Running evaluation for mask stride value: {msv} ###")
        run_eval(config=config, *args, **kwargs)
        if len(mask_stride_values) > 1:
            log(f"### Finished evaluation for mask stride value: {msv} ###")


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='3D evaluation on H36m.')
    parser.add_argument('--weights', required=True,
                        default=None,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file for model weight initialization.")
    parser.add_argument('--config', required=False,
                        default=None,
                        metavar="/path/to/config.json",
                        help="Path to the config file. Overwrites the default configs in the code.")
    parser.add_argument('--gpu_id', required=False,
                        default=None,
                        metavar="gpu_id",
                        help='Overwrites the GPU_ID from the config',
                        type=str)
    parser.add_argument('--batch_size', required=False,
                        default=None,
                        metavar="batch_size",
                        help='Overwrites the BATCH_SIZE from the config',
                        type=int)
    parser.add_argument('--dataset', required=False,
                        default="./data/data_3d_h36m.npz",
                        metavar="/path/to/h36m/.npz",
                        help='3D pose dataset')
    parser.add_argument('--dataset_2d', required=False,
                        default="./data/data_2d_h36m_cpn_ft_h36m_dbb.npz",
                        metavar="/path/to/2d poses/.npz",
                        help='2D pose dataset')
    parser.add_argument('--test_subset', required=False,
                        default="test",
                        metavar="<name of test subset>",
                        help="Name of the dataset subset to evaluate on")
    parser.add_argument('--action_wise', dest='action_wise', action='store_true')
    parser.add_argument('--frame_wise', dest='action_wise', action='store_false')
    parser.set_defaults(action_wise=True)
    parser.add_argument('--forced_mask_stride', required=False,
                        default=None,
                        metavar="forced_mask_stride",
                        help='Overwrites the MASK_STRIDE from the config',
                        type=int)
    parser.add_argument('--no_learned_upsampling', dest='disable_learned_upsampling', action='store_true')
    parser.set_defaults(disable_learned_upsampling=False)

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log(current_time)
    log("Config: ", args.config)
    log("GPU ID: ", args.gpu_id)
    log("Batch size: ", args.batch_size)
    log("Dataset: ", args.dataset)
    log("Dataset 2D: ", args.dataset_2d)
    log("Test subset:", args.test_subset)
    log("Action-wise:", args.action_wise)
    log("Weights:", args.weights)
    if args.disable_learned_upsampling:
        log("Disable learned upsampling:", args.disable_learned_upsampling)
    log("Forced mask stride:", args.forced_mask_stride)

    # Make absolute paths
    args.dataset = path_utils.expandpath(args.dataset)
    args.dataset_2d = path_utils.expandpath(args.dataset_2d)
    if args.config is not None:
        args.config = path_utils.expandpath(args.config)
    if args.weights is not None:
        args.weights = path_utils.expandpath(args.weights)

    # Configuration
    config = UpliftUpsampleConfig(config_file=args.config)
    assert config.ARCH == "UpliftUpsampleTransformer"
    if args.forced_mask_stride is not None:
        log(f"Setting mask stride to fixed value: {args.forced_mask_stride}")
        config.MASK_STRIDE = args.forced_mask_stride

    if args.gpu_id is not None:
        assert args.gpu_id.isalnum()
        config.GPU_ID = int(args.gpu_id)
    if args.batch_size is not None:
        config.BATCH_SIZE = int(args.batch_size)
    if args.disable_learned_upsampling:
        if config.MASK_STRIDE is not None:
            log("WARNING: Disabling learned upsampling. Will use pure bi-linear upsampling.")
            config.EVAL_DISABLE_LEARNED_UPSAMPLING = True

    # Print config
    config.display()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_ID)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) == 1
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            log(e)

    dataset_name = "h36m"
    # Run evaluation for each mask stride value
    run_eval_multi_mask_stride(config=config, dataset_name=dataset_name,
                               dataset_path=args.dataset, dataset2d_path=args.dataset_2d, test_subset=args.test_subset,
                               weights_path=args.weights,
                               action_wise=args.action_wise)

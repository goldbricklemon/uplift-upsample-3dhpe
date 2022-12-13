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
import copy
import time
import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from common.utils import path_utils
from common.utils import time_formatting
from common.utils.metric_history import MetricHistory
from common.net.uplift_upsample_transformer_config import UpliftUpsampleConfig
from common.dataset.keypoint_order import H36MOrder17P
from common.dataset import h36m_splits
from common.dataset.amass_dataset import AMASSDataset
from common.net.uplift_upsample_transformer_constructor import build_uplift_upsample_transformer
from common.utils import schedules
from common.utils import losses_3d
from common.utils import weight_io
from common.dataset import action_wise_eval
from common.dataset.uplifiting_dataset import load_dataset_and_2d_poses, filter_and_subsample_dataset, \
    H36mSequenceGenerator, AMASSSequenceGenerator, tf_world_to_cam_and_2d
import eval


def log(*args):
    print(*args)
    sys.stdout.flush()


def create_h36m_datasets(h36_path, dataset_2d_path, config, train_subset, val_subset, shuffle_seed=0):
    # Build h36m dataset
    dataset_3d, poses_2d_dataset = load_dataset_and_2d_poses(dataset_path=h36_path, poses_2d_path=dataset_2d_path,
                                                             verbose=True)
    train_dataset, val_dataset, val_batches = None, None, None
    for split, selection in zip(["train", "val"], [train_subset, val_subset]):
        if selection is not None:
            # The dataset is subsampled to every Nth frame (i.e. a sequence is extracted at every Nth frame)
            # The frame rate is not changed, however!
            subsample = config.DATASET_TRAIN_3D_SUBSAMPLE_STEP if split == "train" else config.DATASET_VAL_3D_SUBSAMPLE_STEP
            shuffle = split == "train"
            stride_mask_rand_shift = config.STRIDE_MASK_RAND_SHIFT and split == "train"
            subjects = h36m_splits.subjects_by_split[selection]
            actions = "*"

            camera_params, poses_3d, poses_2d, frame_names, \
            sequence_subjects, sequence_actions, sequence_frame_rates = filter_and_subsample_dataset(dataset=dataset_3d,
                                                                                                     poses_2d=poses_2d_dataset,
                                                                                                     subjects=subjects,
                                                                                                     action_filter=actions,
                                                                                                     downsample=1,
                                                                                                     image_base_path=h36_path,
                                                                                                     verbose=True)
            # # NOTE: For now, we perform root shift here
            # for s_i in range(len(poses_3d)):
            #     poses_3d[s_i] = poses_3d[s_i][:, :, :] - poses_3d[s_i][:, [config.ROOT_KEYTPOINT], :]

            do_flip = split == "train" and config.AUGM_FLIP_PROB > 0
            generator = H36mSequenceGenerator(poses_3d, poses_2d,
                                              camera_params=camera_params,
                                              subjects=sequence_subjects,
                                              actions=sequence_actions,
                                              frame_rates=sequence_frame_rates,
                                              split=split,
                                              seq_len=config.SEQUENCE_LENGTH,
                                              target_frame_rate=50,
                                              subsample=subsample,
                                              stride=config.SEQUENCE_STRIDE,
                                              padding_type=config.PADDING_TYPE,
                                              flip_augment=do_flip,
                                              in_batch_augment=config.IN_BATCH_AUGMENT,
                                              flip_lr_indices=config.AUGM_FLIP_KEYPOINT_ORDER,
                                              mask_stride=config.MASK_STRIDE,
                                              stride_mask_align_global=False,
                                              rand_shift_stride_mask=stride_mask_rand_shift,
                                              shuffle=shuffle,
                                              seed=shuffle_seed)
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

            if split == "train":
                dataset = dataset.repeat()
                dataset = dataset.batch(config.BATCH_SIZE)
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                train_dataset = dataset
            else:
                if config.VALIDATION_EXAMPLES < 0:
                    config.VALIDATION_EXAMPLES = len(generator)
                assert config.VALIDATION_EXAMPLES <= len(generator)

                val_batches = int(np.ceil(config.VALIDATION_EXAMPLES / config.BATCH_SIZE))
                # Repeat once such that the last fractional batch can be extracted
                dataset = dataset.repeat(2)
                dataset = dataset.batch(config.BATCH_SIZE)
                dataset = dataset.take(val_batches)
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                val_dataset = dataset

    return train_dataset, val_dataset, val_batches


def create_amass_datasets(amass_path, h36_path, config: UpliftUpsampleConfig, train_subset, val_subset,
                          target_frame_rate, shuffle_seed=0):
    # Build amass h36m dataset
    h36m_cameras = None

    train_dataset, val_dataset, val_batches = None, None, None
    for split, selection in zip(["train", "val"], [train_subset, val_subset]):
        if selection is not None:
            log(f"Loading AMASS dataset for split {selection}")
            amass_dataset = AMASSDataset(path=amass_path, h36m_path=h36_path, split=selection,
                                         h36m_cameras=h36m_cameras)
            # Cache cameras to avoid reloading the entire h36m dataset
            h36m_cameras = amass_dataset.cameras()

            # The dataset is subsampled to every Nth frame (i.e. a sequence is extracted at every Nth frame)
            # The frame rate is not changed, however!
            stride = config.DATASET_TRAIN_3D_SUBSAMPLE_STEP if split == "train" else config.DATASET_VAL_3D_SUBSAMPLE_STEP
            shuffle = split == "train"
            stride_mask_rand_shift = config.STRIDE_MASK_RAND_SHIFT and split == "train"
            do_flip = split == "train" and config.AUGM_FLIP_PROB > 0
            generator = AMASSSequenceGenerator(amass_dataset=amass_dataset,
                                               seq_len=config.SEQUENCE_LENGTH,
                                               target_frame_rate=target_frame_rate,
                                               subsample=stride,
                                               stride=config.SEQUENCE_STRIDE,
                                               padding_type=config.PADDING_TYPE,
                                               flip_augment=do_flip,
                                               in_batch_augment=config.IN_BATCH_AUGMENT,
                                               flip_lr_indices=H36MOrder17P.flip_lr_indices(),
                                               mask_stride=config.MASK_STRIDE,
                                               stride_mask_align_global=False,
                                               rand_shift_stride_mask=stride_mask_rand_shift,
                                               shuffle=shuffle,
                                               seed=shuffle_seed)

            log(f"Sequences: {len(generator)}")

            output_sig = (tf.TensorSpec(shape=(config.SEQUENCE_LENGTH, config.NUM_KEYPOINTS, 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(18,), dtype=tf.float32),
                          tf.TensorSpec(shape=(config.SEQUENCE_LENGTH,), dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.int32),
                          tf.TensorSpec(shape=(), dtype=tf.int32),
                          tf.TensorSpec(shape=(), dtype=tf.int32),
                          tf.TensorSpec(shape=(config.SEQUENCE_LENGTH,), dtype=tf.bool),
                          )
            dataset = tf.data.Dataset.from_generator(generator.next_epoch_iterator, output_signature=output_sig)
            dataset = dataset.map(tf_world_to_cam_and_2d, num_parallel_calls=tf.data.AUTOTUNE)

            if split == "train":
                dataset = dataset.repeat()
                dataset = dataset.batch(config.BATCH_SIZE)
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                train_dataset = dataset
            else:
                if config.VALIDATION_EXAMPLES < 0:
                    config.VALIDATION_EXAMPLES = len(generator)
                assert config.VALIDATION_EXAMPLES <= len(generator)

                val_batches = int(np.ceil(config.VALIDATION_EXAMPLES / config.BATCH_SIZE))
                # Repeat once such that the last fractional batch can be extracted
                dataset = dataset.repeat(2)
                dataset = dataset.batch(config.BATCH_SIZE)
                dataset = dataset.take(val_batches)
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                val_dataset = dataset

    return train_dataset, val_dataset, val_batches


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='2D-to-3D uplifting training for strided poseformer.')
    parser.add_argument('--config', required=False,
                        default=None,
                        metavar="/path/to/config.json",
                        help="Path to the config file. Overwrites the default configs in the code.")
    parser.add_argument('--gpu_id', required=False,
                        default=None,
                        metavar="gpu_id",
                        help='Overwrites the GPU_ID from the config',
                        type=str)
    parser.add_argument('--dataset', required=False,
                        default="h36m",
                        metavar="{h36m, amass}",
                        help='Dataset used for training')
    parser.add_argument('--dataset_val', required=False,
                        default=None,
                        metavar="{h36m, amass}",
                        help='Dataset used for validation')
    parser.add_argument('--h36m_path', required=False,
                        default="./data/data_3d_h36m.npz",
                        metavar="/path/to/h36m/",
                        help='Directory of the H36m dataset')
    parser.add_argument('--amass_path', required=False,
                        default=None,
                        metavar="/path/to/amass/",
                        help='Directory of the AMASS dataset')
    parser.add_argument('--amass_frame_rate', required=False,
                        default="50",
                        metavar="<r>",
                        help='Target frame rate for amass training')
    parser.add_argument('--dataset_2d_path', required=False,
                        default="./data/data_2d_h36m_cpn_ft_h36m_dbb.npz",
                        metavar="/path/to/2d poses/",
                        help='2D pose dataset')
    parser.add_argument('--train_subset', required=False,
                        default="train",
                        metavar="<name of train subset>",
                        help="Name of the dataset subset to train on")
    parser.add_argument('--val_subset', required=False,
                        default="val",
                        metavar="<name of val subset>",
                        help="Name of the dataset subset to validate on\
                                  pass an empty string or \"none\" to disable validation.")
    parser.add_argument('--test_subset', required=False,
                        default=None,
                        metavar="<name of test subset>",
                        help="Name of the dataset subset to test on\
                                  pass an empty string or \"none\" to disable test evaluation.")
    parser.add_argument('--weights', required=False,
                        default=None,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file for model weight initialization.")
    parser.add_argument('--continue_training', required=False,
                        default=False,
                        metavar="<True|False>",
                        help="Try to continue a previously started training, \
                                    mainly loading the weights, optimizer state and epoch number of the latest epoch.")
    parser.add_argument('--out_dir', required=True,
                        metavar="/path/to/output_directory",
                        help='Logs and checkpoint directory. Also used to search for checkpoints if continue_training is set.')

    args = parser.parse_args()
    args.continue_training = args.continue_training not in [False, "False", "false", "f", "n", "0"]
    args.val_subset = None if args.val_subset in ["none", "None", "", 0] else args.val_subset
    args.test_subset = None if args.test_subset in ["none", "None", "", 0] else args.test_subset
    args.dataset = args.dataset.lower()
    args.dataset_val = args.dataset_val.lower() if args.dataset_val is not None else None
    val_dataset_name = args.dataset if args.dataset_val is None else args.dataset_val
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log(current_time)
    log("Config: ", args.config)
    log("GPU ID: ", args.gpu_id)
    log("Dataset: ", args.dataset)
    log("Dataset Val: ", args.dataset_val)
    log("Dataset H36m: ", args.h36m_path)
    log("Dataset AMASS: ", args.amass_path)
    log("AMASS frame rate: ", args.amass_frame_rate)
    log("Dataset 2D: ", args.dataset_2d_path)
    log("Train subset:", args.train_subset)
    log("Val subset:", args.val_subset)
    log("Test subset:", args.test_subset)
    log("Weights:", args.weights)
    log("Continue Training: ", args.continue_training)
    log("Output directory ", args.out_dir)

    assert args.dataset in ["h36m", "amass"]
    assert args.dataset_val in [None, "h36m", "amass"]
    # Make absolute paths
    if args.dataset in ["h36m"] or args.dataset_val in ["h36m"]:
        assert args.dataset_2d_path is not None
    elif args.dataset == "amass" or args.dataset_val == "amass":
        assert args.amass_path is not None
        args.amass_frame_rate = int(args.amass_frame_rate)
    else:
        raise ValueError(f"{args.dataset} is not included in supported datasets.")

    args.h36m_path = path_utils.expandpath(args.h36m_path)
    if args.amass_path is not None:
        args.amass_path = path_utils.expandpath(args.amass_path)
    if args.dataset_2d_path is not None:
        args.dataset_2d_path = path_utils.expandpath(args.dataset_2d_path)
    if args.config is not None:
        args.config = path_utils.expandpath(args.config)
    if args.weights is not None:
        args.weights = path_utils.expandpath(args.weights)

    args.out_dir = path_utils.expandpath(args.out_dir)
    # Create output directory
    path_utils.mkdirs(args.out_dir)

    # Resolve weight path
    args.weights = weight_io.resolve_weight_selector(args.weights)

    # Configuration
    config = UpliftUpsampleConfig(config_file=args.config)
    assert config.ARCH == "UpliftUpsampleTransformer"
    if args.gpu_id is not None:
        assert args.gpu_id.isalnum()
        config.GPU_ID = int(args.gpu_id)

    if val_dataset_name not in ["h36m"] and config.BEST_CHECKPOINT_METRIC is not None:
        config.BEST_CHECKPOINT_METRIC = config.BEST_CHECKPOINT_METRIC.replace("AW-", "")

    # TODO: Set flip order in config.json, not in code
    config.AUGM_FLIP_KEYPOINT_ORDER = H36MOrder17P.flip_lr_indices()

    # Dump complete config to json file (for archiving)
    if args.config:
        split = os.path.split(args.config)
        split_ext = os.path.splitext(split[1])
        out_path = os.path.join(args.out_dir, split_ext[0] + "_complete.json")
    else:
        out_path = os.path.join(args.out_dir, "config_complete.json")

    config.dump(config_file=out_path)

    # Print config
    config.display()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_ID)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) == 1
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            log(e)

    val_subset_name = None if args.dataset_val is not None else args.val_subset
    if args.dataset == "h36m":
        train_dataset, val_dataset, val_batches = create_h36m_datasets(h36_path=args.h36m_path,
                                                                       dataset_2d_path=args.dataset_2d_path,
                                                                       config=config,
                                                                       train_subset=args.train_subset,
                                                                       val_subset=val_subset_name,
                                                                       shuffle_seed=config.SHUFFLE_SEED)
    else:
        train_dataset, val_dataset, val_batches = create_amass_datasets(amass_path=args.amass_path,
                                                                        h36_path=args.h36m_path,
                                                                        config=config,
                                                                        train_subset=args.train_subset,
                                                                        val_subset=val_subset_name,
                                                                        target_frame_rate=args.amass_frame_rate,
                                                                        shuffle_seed=config.SHUFFLE_SEED)

    if args.dataset_val is not None:
        if args.dataset_val == "h36m":
            _, val_dataset, val_batches = create_h36m_datasets(h36_path=args.h36m_path,
                                                               dataset_2d_path=args.dataset_2d_path,
                                                               config=config,
                                                               train_subset=None,
                                                               val_subset=args.val_subset,
                                                               shuffle_seed=config.SHUFFLE_SEED)
        else:
            _, val_dataset, val_batches = create_amass_datasets(amass_path=args.amass_path,
                                                                h36_path=args.h36m_path,
                                                                config=config,
                                                                train_subset=None,
                                                                val_subset=args.val_subset,
                                                                target_frame_rate=args.amass_frame_rate,
                                                                shuffle_seed=config.SHUFFLE_SEED)

    print("val batches", val_batches)
    # Build model, optimizer, checkpoint
    tf_device = "/gpu:0"
    with tf.device(tf_device):
        model = build_uplift_upsample_transformer(config=config)
        if args.weights is not None:
            log(f"Loading weights from {args.weights}")
            weight_io.load_weights_with_callback(model, filepath=args.weights, skip_mismatch=False)
        val_model = model

        # Keep an exponential moving average of the actual model
        if config.EMA_ENABLED is True:
            log("Cloning EMA model.")
            ema_model = build_uplift_upsample_transformer(config=config)
            # Copy weights
            ema_model.set_weights(model.get_weights())
            val_model = ema_model

        # Create LR Schedule
        lr_schedule = schedules.scheduler_by_name(config.SCHEDULE)(**config.SCHEDULE_PARAMS)

        log(f"Using {config.OPTIMIZER} optimizer")
        if config.OPTIMIZER == "AdamW":
            wd_params = copy.deepcopy(config.SCHEDULE_PARAMS)
            wd_params["initial_learning_rate"] = config.WEIGHT_DECAY
            log(wd_params)
            wd_schedule = schedules.scheduler_by_name(config.SCHEDULE)(**wd_params)
            optimizer = tfa.optimizers.AdamW(weight_decay=wd_schedule,
                                             learning_rate=lr_schedule,
                                             epsilon=1e-8,
                                             **config.OPTIMIZER_PARAMS)
        elif config.OPTIMIZER == "Adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, **config.OPTIMIZER_PARAMS)
        else:
            raise ValueError(config.OPTIMIZER)
        ckp_dict = {"optimizer": optimizer,
                    "model":     model}
        if config.EMA_ENABLED is True:
            ckp_dict["ema_model"] = ema_model
        checkpoint = tf.train.Checkpoint(**ckp_dict)
        checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
        path_utils.mkdirs(checkpoint_dir)
        checkpoint_template = os.path.join(checkpoint_dir, "cp_{:04d}.ckpt")

        initial_epoch = 1
        if args.continue_training:
            ckp_path = tf.train.latest_checkpoint(checkpoint_dir)
            assert ckp_path is not None, "Cant find checkpoint to continue training"
            log(f"Restoring checkpoint from {ckp_path}")
            checkpoint.restore(ckp_path)
            initial_epoch = int(os.path.splitext(ckp_path)[0][-4:]) + 1
            log(f"Will continue training from epoch {initial_epoch}")

    global_step = (initial_epoch - 1) * config.STEPS_PER_EPOCH

    # Metrics and Tensorboard
    train_epoch_loss = keras.metrics.Mean()
    val_run_loss = keras.metrics.Mean()

    tb_log_dir = os.path.join(args.out_dir, "tb_" + current_time)
    tb_writer = tf.summary.create_file_writer(tb_log_dir)

    prev_best_weights_path = None
    last_weights_path = None

    metric_hist = MetricHistory()
    metrics = ["loss", "AMPJPE", "MPJPE", "NMPJPE", "PAMPJPE"]
    higher_is_better = [False, False, False, False, False]
    if val_dataset_name == "h36m":
        metrics += ["AW-MPJPE", "AW-NMPJPE", "AW-PAMPJPE"]
        higher_is_better += [False, False, False]
    for m, h in zip(metrics,
                    higher_is_better):
        metric_hist.add_metric(m, higher_is_better=h)

    if config.BEST_CHECKPOINT_METRIC is not None:
        assert config.BEST_CHECKPOINT_METRIC in metrics


    @tf.function
    def train_step(keypoints2d, keypoints3d, stride_masks, cams, ema_decay):
        absolute_keypoints3d = keypoints3d
        keypoints3d = keypoints3d - keypoints3d[:, :, config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
        mid_index = config.SEQUENCE_LENGTH // 2
        central_keypoints_3d = keypoints3d[:, mid_index]

        model_input = keypoints2d
        if model.has_strided_input:
            # Stride mask is 1 on valid (i.e. non-masked) indices !!!
            masked_keypoints2d = keypoints2d * tf.cast(stride_masks[:, :, tf.newaxis, tf.newaxis], dtype=tf.float32)
            model_input = [masked_keypoints2d, stride_masks]

        with tf.GradientTape() as tape:
            pred_keypoints_3d, pred_keypoints_3d_central = model(model_input, training=True)
            # central_loss is: (B, K)
            central_loss = losses_3d.tf_mpjpe(pred=pred_keypoints_3d_central, gt=central_keypoints_3d)
            # Aggregate loss over keypoints and batch
            central_loss = tf.math.reduce_sum(central_loss) / (config.BATCH_SIZE * config.NUM_KEYPOINTS)

            if config.TEMPORAL_TRANSFORMER_BLOCKS > 0:
                # sequence_loss is: (B, N, K)
                sequence_loss = losses_3d.tf_mpjpe(pred=pred_keypoints_3d, gt=keypoints3d)
                # Aggregate loss over keypoints, sequence and batch
                sequence_loss = tf.math.reduce_sum(sequence_loss) / (
                        config.BATCH_SIZE * config.SEQUENCE_LENGTH * config.NUM_KEYPOINTS)

                loss = (config.LOSS_WEIGHT_CENTER * central_loss) + (config.LOSS_WEIGHT_SEQUENCE * sequence_loss)
            else:
                # Fallback without temporal transformer blocks: disable sequence loss
                loss = (config.LOSS_WEIGHT_CENTER + config.LOSS_WEIGHT_SEQUENCE) * central_loss

        trainable_variables = model.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        train_epoch_loss.update_state(loss)

        if config.EMA_ENABLED is True:
            for w, ema_w in zip(model.weights, ema_model.weights):
                ema_w.assign_sub((1 - ema_decay) * (ema_w - w))

        return loss


    @tf.function
    def val_step(keypoints2d, keypoints3d, stride_masks):
        keypoints3d = keypoints3d - keypoints3d[:, :, config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
        mid_index = config.SEQUENCE_LENGTH // 2
        central_keypoints_3d = keypoints3d[:, mid_index]

        model_input = keypoints2d
        if model.has_strided_input:
            masked_keypoints2d = keypoints2d * tf.cast(stride_masks[:, :, tf.newaxis, tf.newaxis], dtype=tf.float32)
            model_input = [masked_keypoints2d, stride_masks]

        pred_keypoints_3d, pred_keypoints_3d_central = val_model(model_input, training=False)
        # central_loss is: (B, K)
        central_loss = losses_3d.tf_mpjpe(pred=pred_keypoints_3d_central, gt=central_keypoints_3d)
        # Aggregate loss over keypoints and batch
        central_loss = tf.math.reduce_sum(central_loss) / (config.BATCH_SIZE * config.NUM_KEYPOINTS)
        loss = central_loss

        if config.TEMPORAL_TRANSFORMER_BLOCKS > 0:
            # sequence_loss is: (B, N, K)
            sequence_loss = losses_3d.tf_mpjpe(pred=pred_keypoints_3d, gt=keypoints3d)
            # Aggregate loss over keypoints, sequence and batch
            sequence_loss = tf.math.reduce_sum(sequence_loss) / (
                    config.BATCH_SIZE * config.SEQUENCE_LENGTH * config.NUM_KEYPOINTS)
            loss = central_loss + sequence_loss
        else:
            loss = central_loss
        val_run_loss.update_state(loss)

        return pred_keypoints_3d_central


    # Train loop
    train_dataset_iter = iter(train_dataset)
    ema_decay = tf.constant(0.0, dtype=tf.float32)
    mid_index = config.SEQUENCE_LENGTH // 2
    epoch_duration = 0.
    # Epochs use 1-based index
    for epoch in range(initial_epoch, config.EPOCHS + 1):
        train_epoch_loss.reset_states()
        epoch_start = time.time()
        log(f"## EPOCH {epoch} / {config.EPOCHS}")
        # (Global) Steps use 0-based index
        for iteration in range(config.STEPS_PER_EPOCH):
            tick = time.time()
            if config.EMA_ENABLED:
                ema_decay = tf.constant(min(config.EMA_DECAY, (1.0 + global_step) / (10.0 + global_step)),
                                        dtype=tf.float32)
            sequences_3d, sequences_2d, sequences_mask, sequence_camera_params, _, _, _, stride_masks = next(
                train_dataset_iter)
            loss = train_step(keypoints2d=sequences_2d, keypoints3d=sequences_3d,
                              stride_masks=stride_masks, cams=sequence_camera_params,
                              ema_decay=ema_decay)

            tock = time.time()
            step_duration = tock - tick
            epoch_duration = tock - epoch_start

            if iteration % 10 == 0:
                eta = ((config.STEPS_PER_EPOCH - iteration - 1) / (iteration + 1)) * epoch_duration
                eta_string = time_formatting.format_time(eta)
                log(f"{iteration}/{config.STEPS_PER_EPOCH} @ Epoch {epoch} "
                    f"(Step {step_duration:.3f}s, ETA {eta_string}): "
                    f"Mean loss {float(train_epoch_loss.result()):.6f}")

            global_step += 1

        # Checkpoint
        if epoch % config.CHECKPOINT_INTERVAL == 0:
            save_path = checkpoint.save(checkpoint_template.format(epoch))
            log("Saving checkpoint to ", save_path)

        if config.STEPS_PER_EPOCH > 0:
            epoch_duration_string = time_formatting.format_time(epoch_duration)
            mean_step_duration_string = epoch_duration / config.STEPS_PER_EPOCH
            log(f"Finished epoch {epoch} in {epoch_duration_string}, {mean_step_duration_string:.3f}s/step")
            with tb_writer.as_default():
                tf.summary.scalar('train/loss', train_epoch_loss.result(), step=epoch)
                tf.summary.scalar('train/LR', float(optimizer._decayed_lr(var_dtype=tf.float32)), step=epoch)
                if config.OPTIMIZER == "AdamW":
                    tf.summary.scalar('train/WD', float(optimizer._decayed_wd(var_dtype=tf.float32)), step=epoch)
                tf.summary.scalar('train/step_duration', epoch_duration / config.STEPS_PER_EPOCH, step=epoch)

        if epoch % config.VALIDATION_INTERVAL == 0 and args.val_subset is not None:
            log(f"Running validation on {config.VALIDATION_EXAMPLES} examples")
            val_start = time.time()
            val_run_loss.reset_states()
            val_gt_keypoints3d = list()
            val_pred_keypoints3d = list()
            val_gt_subjects = list()
            val_gt_actions = list()
            examples = 0
            for b_i, (
                    val_sequences_3d, val_sequences_2d, val_sequences_mask,
                    val_sequence_camera_params, val_sequence_subjects, val_sequence_actions, _,
                    val_stride_masks) in enumerate(
                val_dataset):
                pred_keypoints3d = val_step(keypoints2d=val_sequences_2d, keypoints3d=val_sequences_3d,
                                            stride_masks=val_stride_masks)
                if config.EVAL_FLIP is True:
                    flipped_sequences_2d = val_sequences_2d
                    flipped_sequences_2d = tf.concat([flipped_sequences_2d[:, :, :, :1] * -1.,
                                                      flipped_sequences_2d[:, :, :, 1:]], axis=-1)
                    flipped_sequences_2d = tf.gather(flipped_sequences_2d, indices=config.AUGM_FLIP_KEYPOINT_ORDER,
                                                     axis=2)
                    flipped_sequences_3d = val_sequences_3d
                    flipped_sequences_3d = tf.concat([flipped_sequences_3d[:, :, :, :1] * -1.,
                                                      flipped_sequences_3d[:, :, :, 1:]], axis=-1)
                    flipped_sequences_3d = tf.gather(flipped_sequences_3d, indices=config.AUGM_FLIP_KEYPOINT_ORDER,
                                                     axis=2)
                    flipped_pred_keypoints_3d = val_step(keypoints2d=flipped_sequences_2d,
                                                         keypoints3d=flipped_sequences_3d,
                                                         stride_masks=val_stride_masks)
                    flipped_pred_keypoints_3d = tf.concat([flipped_pred_keypoints_3d[:, :, :1] * -1.,
                                                           flipped_pred_keypoints_3d[:, :, 1:]], axis=-1)
                    flipped_pred_keypoints_3d = tf.gather(flipped_pred_keypoints_3d,
                                                          indices=config.AUGM_FLIP_KEYPOINT_ORDER, axis=1)
                    pred_keypoints3d += flipped_pred_keypoints_3d
                    pred_keypoints3d /= 2.

                # Only collect as many examples as needed
                examples_to_include = min(config.BATCH_SIZE, config.VALIDATION_EXAMPLES - examples)
                # Perform root-shift right before metric calculation
                val_sequences_3d = val_sequences_3d - val_sequences_3d[:, :,
                                                      config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
                val_central_keypoints_3d = val_sequences_3d[:, mid_index]
                val_gt_keypoints3d.extend(val_central_keypoints_3d[:examples_to_include].numpy())
                val_pred_keypoints3d.extend(pred_keypoints3d[:examples_to_include].numpy())
                val_gt_subjects.extend(val_sequence_subjects[:examples_to_include].numpy())
                val_gt_actions.extend(val_sequence_actions[:examples_to_include].numpy())
                examples += examples_to_include

            val_gt_keypoints3d = np.stack(val_gt_keypoints3d, axis=0).astype(np.float64)
            # Add dummy valid flag
            val_gt_keypoints3d = np.concatenate([val_gt_keypoints3d, np.ones(val_gt_keypoints3d.shape[:-1] + (1,))],
                                                axis=-1)
            val_pred_keypoints3d = np.stack(val_pred_keypoints3d, axis=0).astype(np.float64)
            val_gt_subjects = np.stack(val_gt_subjects, axis=0)
            val_gt_actions = np.stack(val_gt_actions, axis=0)
            assert b_i == (val_batches - 1)

            if val_dataset_name == "h36m":
                # Run H36m 3D evaluation
                frame_results, action_wise_results, _ = action_wise_eval.h36_action_wise_eval(
                    pred_3d=val_pred_keypoints3d,
                    gt_3d=val_gt_keypoints3d,
                    actions=val_gt_actions,
                    root_index=config.ROOT_KEYTPOINT)
            else:
                frame_results = action_wise_eval.frame_wise_eval(
                    pred_3d=val_pred_keypoints3d,
                    gt_3d=val_gt_keypoints3d,
                    root_index=config.ROOT_KEYTPOINT)

            val_duration = time.time() - val_start
            val_duration_string = time_formatting.format_time(val_duration)

            log(
                f"Finished validation in {val_duration_string}, loss: {float(val_run_loss.result()):.6f}, "
                f"AMPJPE: {frame_results['ampjpe']:.2f}, "
                f"MPJPE: {frame_results['mpjpe']:.2f}, "
                f"NMPJPE: {frame_results['nmpjpe']:.2f}, "
                f"PAMPJPE: {frame_results['pampjpe']:.2f}, "
            )
            if val_dataset_name == "h36m":
                log(
                    f"AW-MPJPE: {action_wise_results['mpjpe']:.2f}, "
                    f"AW-NMPJPE: {action_wise_results['nmpjpe']:.2f}, "
                    f"AW-PAMPJPE: {action_wise_results['pampjpe']:.2f}, "
                )

            with tb_writer.as_default():
                tf.summary.scalar('val/loss', val_run_loss.result(), step=epoch)
                tf.summary.scalar('val/AMPJPE', frame_results['ampjpe'], step=epoch)
                tf.summary.scalar('val/MPJPE', frame_results['mpjpe'], step=epoch)
                tf.summary.scalar('val/NMPJPE', frame_results['nmpjpe'], step=epoch)
                tf.summary.scalar('val/PAMPJPE', frame_results['pampjpe'], step=epoch)
                if val_dataset_name == "h36m":
                    tf.summary.scalar('val/AW-MPJPE', action_wise_results['mpjpe'], step=epoch)
                    tf.summary.scalar('val/AW-NMPJPE', action_wise_results['nmpjpe'], step=epoch)
                    tf.summary.scalar('val/AW-PAMPJPE', action_wise_results['pampjpe'], step=epoch)

            metric_hist.add_data("loss", value=val_run_loss.result(), step=epoch)
            metric_hist.add_data("AMPJPE", value=frame_results['ampjpe'], step=epoch)
            metric_hist.add_data("MPJPE", value=frame_results['mpjpe'], step=epoch)
            metric_hist.add_data("NMPJPE", value=frame_results['nmpjpe'], step=epoch)
            metric_hist.add_data("PAMPJPE", value=frame_results['pampjpe'], step=epoch)
            if val_dataset_name == "h36m":
                metric_hist.add_data("AW-MPJPE", value=action_wise_results['mpjpe'], step=epoch)
                metric_hist.add_data("AW-NMPJPE", value=action_wise_results['nmpjpe'], step=epoch)
                metric_hist.add_data("AW-PAMPJPE", value=action_wise_results['pampjpe'], step=epoch)

            if config.BEST_CHECKPOINT_METRIC is not None and args.val_subset is not None:
                # Save best checkpoint as .h5
                best_value, best_epoch = metric_hist.best_value(config.BEST_CHECKPOINT_METRIC)
                if best_epoch == epoch:
                    print(
                        f"Saving currently best checkpoint @ epoch {best_epoch} ({config.BEST_CHECKPOINT_METRIC}: {best_value}) as .h5:")
                    weights_path = os.path.join(checkpoint_dir, f"best_weights_{best_epoch:04d}.h5")
                    print(weights_path)
                    val_model.save_weights(weights_path)

                    if prev_best_weights_path is not None:
                        os.remove(prev_best_weights_path)

                    prev_best_weights_path = weights_path

        print(f"Saving last checkpoint @ epoch {epoch} as .h5:")
        if last_weights_path is not None:
            os.remove(last_weights_path)

        last_weights_path = os.path.join(checkpoint_dir, f"last_weights_{epoch:04d}.h5")
        print(last_weights_path)
        val_model.save_weights(last_weights_path)

    del train_dataset_iter
    del train_dataset
    del val_dataset

    tb_writer.close()

    if args.val_subset is not None:
        log(f"Best checkpoint results:")
        if config.BEST_CHECKPOINT_METRIC is not None:
            metric_hist.print_all_for_best_metric(metric=config.BEST_CHECKPOINT_METRIC)
        else:
            metric_hist.print_best()

    if args.test_subset is not None and val_dataset_name in ["h36m"]:
        if config.BEST_CHECKPOINT_METRIC is not None and args.val_subset is not None:
            print("Eval best weights")
            eval_weights_path = prev_best_weights_path
        else:
            print("Eval last weights")
            eval_weights_path = last_weights_path

        eval.run_eval_multi_mask_stride(config=config,
                                        dataset_name=val_dataset_name,
                                        dataset_path=args.h36m_path,
                                        dataset2d_path=args.dataset_2d_path,
                                        test_subset=args.test_subset,
                                        weights_path=eval_weights_path,
                                        model=None,
                                        action_wise=True)

    log("Done.")

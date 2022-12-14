# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import numpy as np
import sys

from common.dataset import h36m_splits
from common.dataset import metrics as h36metrics


def h36_action_wise_eval(pred_3d, gt_3d, actions, root_index):
    action_set = h36m_splits.renamed_actions
    metrics = ["mpjpe", "nmpjpe", "pampjpe"]
    per_action_results = {}

    average = lambda a: np.mean(a[a >= 0])

    # Run H36m 3D evaluation
    frame_mpjpe = h36metrics.mpjpe(pred=pred_3d, gt=gt_3d,
                             root_index=root_index, normalize=False) * 1000.
    frame_nmpjpe = h36metrics.nmpjpe(pred=pred_3d, gt=gt_3d,
                               root_index=root_index, alignment="root", normalize=False) * 1000.
    frame_pampjpe = h36metrics.pmpjpe(pred=pred_3d, gt=gt_3d, normalize=False) * 1000.

    for a_i, action_name in enumerate(action_set):
        selector = np.where(actions == a_i)
        mpjpe = average(frame_mpjpe[selector])
        nmpjpe = average(frame_nmpjpe[selector])
        pampjpe = average(frame_pampjpe[selector])

        out_dict = {}
        for name, value in zip(metrics, [mpjpe, nmpjpe, pampjpe]):
            out_dict[name] = value
        per_action_results[action_name] = out_dict

    frame_results = {}
    mpjpe = average(frame_mpjpe)
    nmpjpe = average(frame_nmpjpe)
    pampjpe = average(frame_pampjpe)
    for name, value in zip(metrics, [mpjpe, nmpjpe, pampjpe]):
        frame_results[name] = value

    average_results = {}
    for metric in metrics:
        value = np.mean([d[metric] for _, d in per_action_results.items()])
        average_results[metric] = value

    return frame_results, average_results, per_action_results


def frame_wise_eval(pred_3d, gt_3d, root_index):
    metrics = ["mpjpe", "nmpjpe", "pampjpe"]
    # Run H36m 3D evaluation
    frame_mpjpe = h36metrics.mpjpe(pred=pred_3d, gt=gt_3d,
                             root_index=root_index, normalize=False) * 1000.
    frame_nmpjpe = h36metrics.nmpjpe(pred=pred_3d, gt=gt_3d,
                               root_index=root_index, alignment="root", normalize=False) * 1000.
    frame_pampjpe = h36metrics.pmpjpe(pred=pred_3d, gt=gt_3d, normalize=False) * 1000.

    frame_results = {}
    average = lambda a: np.mean(a[a >= 0])
    mpjpe = average(frame_mpjpe)
    nmpjpe = average(frame_nmpjpe)
    pampjpe = average(frame_pampjpe)
    for name, value in zip(metrics, [mpjpe, nmpjpe, pampjpe]):
        frame_results[name] = value
    return frame_results


def interpolate_between_keyframes(pred3d, frame_indices, keyframe_stride):
    interp3d = np.copy(pred3d)
    last_keyframe = None
    keyframes = np.equal(np.mod(frame_indices, keyframe_stride), 0)
    for i, (f, is_keyframe) in enumerate(zip(frame_indices, keyframes)):

        if i > 0 and f <= frame_indices[i-1]:
            last_keyframe = None

        if is_keyframe:
            if last_keyframe is not None:
                for k in range(last_keyframe + 1, i):
                    # Interpolate
                    d_left = k - last_keyframe
                    d_right = i - k
                    d_sum = d_left + d_right
                    w_left = d_right / d_sum
                    w_right = d_left / d_sum
                    interp3d[k] = (pred3d[last_keyframe] * w_left) + (pred3d[i] * w_right)

            last_keyframe = i
        else:
            interp3d[i] = pred3d[last_keyframe]

    return interp3d, keyframes


def compute_and_log_metrics(pred3d, gt3d, actions, root_index, action_wise):

    def log(*args):
        print(*args)
        sys.stdout.flush()

    log("Computing metrics:")
    metrics = ["mpjpe", "nmpjpe", "pampjpe"]
    frame_results, average_results, per_action_results = h36_action_wise_eval(
        pred_3d=pred3d,
        gt_3d=gt3d,
        actions=actions,
        root_index=root_index)

    log("Frame-wise evaluation:")
    for metric_name in metrics:
        log(f"{metric_name.upper()}: {frame_results[metric_name]:.3f}")
    log("")

    if action_wise is True:
        for action_name in sorted(per_action_results.keys()):
            res = per_action_results[action_name]
            log(f'Results for "{action_name}"')
            for metric_name in metrics:
                log(f"{metric_name.upper()}: {res[metric_name]:.3f}")

        log(f"Total action-wise evaluation results:")
        for metric_name in metrics:
            log(f"{metric_name.upper()}: {average_results[metric_name]:.3f}")

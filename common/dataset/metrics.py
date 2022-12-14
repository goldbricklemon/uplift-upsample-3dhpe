# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import numpy as np


def mpjpe(pred, gt, root_index, normalize=True):
    """
    Calculates the root-aligned MPJPE i.e.
    1/N * 1\K * \sum_i \sum_j ( || (p_i,j - p_i_root) - (gt_i,j - gt_i_root) ||_2 )
    :param pred: np.array (B, K, 3). 3d pose predictions for B examples, K keypoints, in (x,y,z) format.
    :param gt: np.array (B, K, 4). 3d pose gt in (x,y,z,v) format, where v is the valid flag.
    Only keypoints with v > 0 will be included in the metric.
    :param root_index: index of the root keypoint.
    :param normalize: If True, sum and normalize to obtain a single scalar value.
    If false, return the per-example, per-keypoint JPE. The JPE will be -1 for keypoints with invalid GT.
    :return: float, if normalize=True, np.array (B,K) otherwise.
    """

    gt3d = gt[:, :, :3]
    valid = gt[:, :, 3] > 0
    gt3d = gt3d - gt3d[:, root_index, np.newaxis, :]
    pred3d = pred - pred[:, root_index, np.newaxis, :]
    dist = np.linalg.norm(pred3d - gt3d, ord=2, axis=-1)
    if normalize is False:
        dist = np.where(valid, dist, -1.)
        return dist
    else:
        dist = np.where(valid, dist, 0.)
        norm = float(np.sum(valid > 0.))
        return np.sum(dist) / norm


def nmpjpe(pred, gt, root_index, alignment="root", normalize=True):
    """
    Calculates the *N*ormalized MPJPE w.r.t. position and scale i.e.
    1/N * \sum_i min_s (1\K * \sum_j ( || s*(p_i,j - p_i_root) - (gt_i,j - gt_i_root)||_2 )).
    Alternatively, uses mean-alignment instead of root alignment
    :param pred: np.array (B, K, 3). 3d pose predictions for B examples, K keypoints, in (x,y,z) format.
    :param gt: np.array (B, K, 4). 3d pose gt in (x,y,z,v) format, where v is the valid flag.
    Only keypoints with v > 0 will be included in the metric.
    :param root_index: index of the root keypoint.
    :param alignment: Alignment variant, "mean" or "root"
    :param normalize: If True, sum and normalize to obtain a single scalar value.
    If false, return the per-example, per-keypoint JPE. The JPE will be -1 for keypoints with invalid GT.
    :return: float, if normalize=True, np.array (B,K) otherwise.
    """

    gt3d = gt[:, :, :3]
    valid = gt[:, :, 3] > 0

    if alignment == "mean":
        normalizer = np.sum(valid, axis=1)
        mean_gt3d = np.sum(gt3d * valid[:, :, np.newaxis], axis=1) / normalizer[:, np.newaxis]
        gt3d = gt3d - mean_gt3d[:, np.newaxis, :]

        mean_pred3d = np.sum(pred * valid[:, :, np.newaxis], axis=1) / normalizer[:, np.newaxis]
        pred3d = pred - mean_pred3d[:, np.newaxis, :]

    else:
        # Align by root node
        gt3d = gt3d - gt3d[:, [root_index], :]
        pred3d = pred - pred[:, [root_index], :]

    # Optimal scaling
    pred3d = optimal_scaling(pred3d=pred3d, target3d=gt3d, valid_mask=valid)

    dist = np.linalg.norm(pred3d - gt3d, ord=2, axis=-1)
    if normalize is False:
        dist = np.where(valid, dist, -1.)
        return dist
    else:
        dist = np.where(valid, dist, 0.)
        norm = float(np.sum(valid > 0.))
        return np.sum(dist) / norm


def pmpjpe(pred, gt, normalize=True):
    """
    Calculates the *P*rocustes aligned MPJPE i.e.
    1/N * \sum_i min_t min_s min_R (1\K * \sum_j ( || (s*R*p_i,j) + t - gt_i,j||_2 )).
    :param pred: np.array (B, K, 3). 3d pose predictions for B examples, K keypoints, in (x,y,z) format.
    :param gt: np.array (B, K, 4). 3d pose gt in (x,y,z,v) format, where v is the valid flag.
    Only keypoints with v > 0 will be included in the metric.
    :param normalize: If True, sum and normalize to obtain a single scalar value.
    If false, return the per-example, per-keypoint JPE. The JPE will be -1 for keypoints with invalid GT.
    :return: float, if normalize=True, np.array (B,K) otherwise.
    """

    gt3d = gt[:, :, :3]
    valid = gt[:, :, 3] > 0

    aligned_pred = []
    for p, g in zip(pred, gt3d):
        try:
            _, p_aligned, _, _, _ = compute_similarity_transform(X=g, Y=p, compute_optimal_scale=True)
            aligned_pred.append(p_aligned)
        except np.linalg.LinAlgError as e:
            print("Warning: SVD did not converge during PAMPJPE")
            aligned_pred.append(p)

    aligned_pred = np.stack(aligned_pred, axis=0)

    dist = np.linalg.norm(aligned_pred - gt3d, ord=2, axis=-1)
    if normalize is False:
        dist = np.where(valid, dist, -1.)
        return dist
    else:
        dist = np.where(valid, dist, 0.)
        norm = float(np.sum(valid > 0.))
        return np.sum(dist) / norm


def optimal_scaling(pred3d, target3d, valid_mask):
    masked_target, masked_pred = target3d * valid_mask[:, :, np.newaxis], pred3d * valid_mask[:, :, np.newaxis]
    # Calculate scales
    nom = np.sum(masked_pred[:, :, 0] * masked_target[:, :, 0], axis=1) + \
          np.sum(masked_pred[:, :, 1] * masked_target[:, :, 1], axis=1) + \
          np.sum(masked_pred[:, :, 2] * masked_target[:, :, 2], axis=1)
    denom = np.sum(masked_pred[:, :, 0] * masked_pred[:, :, 0], axis=1) + \
            np.sum(masked_pred[:, :, 1] * masked_pred[:, :, 1], axis=1) + \
            np.sum(masked_pred[:, :, 2] * masked_pred[:, :, 2], axis=1)

    s_opt = nom / denom

    scaled_pred = pred3d * s_opt[:, np.newaxis, np.newaxis]
    return scaled_pred


def compute_similarity_transform(X, Y, compute_optimal_scale=True):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Taken from https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/procrustes.py,
    by Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.
    published on Github under the MIT licence

    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    import numpy as np

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = np.square(X0).sum()
    ssY = np.square(Y0).sum()

    # centered Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - np.square(traceTA)
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c

# Code adapted from: https://github.com/facebookresearch/VideoPose3D
# Original Code: Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
# import torch

from common.dataset.quaternion import np_qrot, np_qinverse


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = np_qinverse(R)  # Invert rotation
    return np_qrot(np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


def camera_to_world(X, R, t):
    return np_qrot(np.tile(R, (*X.shape[:-1], 1)), X) + t


def project_to_2d_linear(X, f, c):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3
    """
    assert X.shape[-1] == 3

    XX = X[..., :2] / X[..., 2:]

    return f*XX + c

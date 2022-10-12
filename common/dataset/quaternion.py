# Code adapted from: https://github.com/facebookresearch/VideoPose3D
# Original Code: Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


def np_qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = np.cross(qvec, v, axis=len(q.shape) - 1)
    uuv = np.cross(qvec, uv, axis=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def np_qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return np.concatenate([w, -xyz], axis=-1)

# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import numpy as np


class H36MOrderFull:
    """
    Helper for the Human3.6M pose definition (original order, including redundant keypoints)
    """

    pelvis = 0
    r_hip = 1
    r_knee = 2
    r_ankle = 3
    r_foot = 4
    r_toes = 5
    l_hip = 6
    l_knee = 7
    l_ankle = 8
    l_foot = 9
    l_toes = 10
    same_as_pelvis = 11
    torso = 12
    neck = 13
    head = 14
    head_top = 15
    same_as_neck = 16
    l_shoulder = 17
    l_elbow = 18
    l_wrist = 19
    same_as_l_wrist = 20
    l_thumb = 21
    l_fingers = 22
    same_as_l_fingers = 23
    same_as_neck_2 = 24
    r_shoulder = 25
    r_elbow = 26
    r_wrist = 27
    same_as_r_wrist = 28
    r_thumb = 29
    r_fingers = 30
    same_as_r_fingers = 31

    num_points = 32

    _indices = [pelvis,
                r_hip, r_knee, r_ankle, r_foot, r_toes,
                l_hip, l_knee, l_ankle, l_foot, l_toes,
                same_as_pelvis,
                torso, neck, head, head_top,
                same_as_neck,
                l_shoulder, l_elbow, l_wrist, same_as_l_wrist, l_thumb, l_fingers, same_as_l_fingers,
                r_shoulder, r_elbow, r_wrist, same_as_r_wrist, r_thumb, r_fingers, same_as_r_fingers,
                ]

    @classmethod
    def indices(cls):
        return cls._indices


    @classmethod
    def to_17p_order(cls):
        """
        Matches the official 17point selection, but in a custom order that is close to MPII Human Pose order
        :return:
        """
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.pelvis,
                cls.neck, cls.torso,
                cls.head, cls.head_top,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                ]

    def __init__(self):
        pass



class H36MOrder:
    """
    Helper for the Human3.6M pose definition (excluding redundant keypoints)
    """

    pelvis = 0
    r_hip = 1
    r_knee = 2
    r_ankle = 3
    r_foot = 4
    r_toes = 5
    l_hip = 6
    l_knee = 7
    l_ankle = 8
    l_foot = 9
    l_toes = 10
    torso = 11
    neck = 12
    head = 13
    head_top = 14
    l_shoulder = 15
    l_elbow = 16
    l_wrist = 17
    l_thumb = 18
    l_fingers = 19
    r_shoulder = 20
    r_elbow = 21
    r_wrist = 22
    r_thumb = 23
    r_fingers = 24

    num_points = 25

    _indices = [pelvis,
                r_hip, r_knee, r_ankle, r_foot, r_toes,
                l_hip, l_knee, l_ankle, l_foot, l_toes,
                torso, neck, head, head_top,
                l_shoulder, l_elbow, l_wrist, l_thumb, l_fingers,
                r_shoulder, r_elbow, r_wrist, r_thumb, r_fingers,
                ]

    _flip_lr_indices = [pelvis,
                        l_hip, l_knee, l_ankle, l_foot, l_toes,
                        r_hip, r_knee, r_ankle, r_foot, r_toes,
                        torso, neck, head, head_top,
                        r_shoulder, r_elbow, r_wrist, r_thumb, r_fingers,
                        l_shoulder, l_elbow, l_wrist, l_thumb, l_fingers,
                        ]

    @classmethod
    def indices(cls):
        return cls._indices

    @classmethod
    def flip_lr_indices(cls):
        return cls._flip_lr_indices

    @classmethod
    def to_15p_order(cls):
        """
        Matches the MPII Human Pose 15point order
        :return:
        """
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.pelvis,
                cls.neck, cls.head_top,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                ]

    @classmethod
    def to_17p_order(cls):
        """
        Matches our 17point order
        :return:
        """
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.pelvis,
                cls.neck, cls.torso,
                cls.head, cls.head_top,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                ]

    def __init__(self):
        pass


class H36MOrder17P:
    """
    Helper for the Human3.6M 17-point pose definition (in our custom order similar to MPII Human Pose)
    """

    r_ankle = 0
    r_knee = 1
    r_hip = 2
    l_hip = 3
    l_knee = 4
    l_ankle = 5
    pelvis = 6
    neck = 7
    torso = 8
    head = 9
    head_top = 10
    r_wrist = 11
    r_elbow = 12
    r_shoulder = 13
    l_shoulder = 14
    l_elbow = 15
    l_wrist = 16

    num_points = 17
    num_bodyparts = 16

    _indices = [r_ankle, r_knee, r_hip,
                l_hip, l_knee, l_ankle,
                pelvis,
                neck,
                torso,
                head, head_top,
                r_wrist, r_elbow, r_shoulder,
                l_shoulder, l_elbow, l_wrist,
                ]

    _flip_lr_indices = [l_ankle, l_knee, l_hip,
                        r_hip, r_knee, r_ankle,
                        pelvis,
                        neck,
                        torso,
                        head, head_top,
                        l_wrist, l_elbow, l_shoulder,
                        r_shoulder, r_elbow, r_wrist,
                        ]

    _bodypart_indices = [[head_top, head], [head, neck],
                         [neck, torso], [torso, pelvis],
                         [neck, r_shoulder],[r_shoulder, r_elbow], [r_elbow, r_wrist],
                         [neck, l_shoulder], [l_shoulder, l_elbow], [l_elbow, l_wrist],
                         [pelvis, r_hip], [r_hip, r_knee], [r_knee, r_ankle],
                         [pelvis, l_hip], [l_hip, l_knee], [l_knee, l_ankle],
                         ]
    _limb_indices = [[head_top, head], [head, neck],
                     [r_shoulder, r_elbow], [r_elbow, r_wrist],
                     [l_shoulder, l_elbow], [l_elbow, l_wrist],
                     [pelvis, r_hip], [r_hip, r_knee], [r_knee, r_ankle],
                     [pelvis, l_hip], [l_hip, l_knee], [l_knee, l_ankle],
                     ]

    _names = ["rank", "rknee", "rhip",
              "lhip", "lknee", "lank",
              "pelv",
              "neck", "torso", "head", "htop",
              "rwri", "relb", "rsho",
              "lsho", "lelb", "lwrit"
              ]

    _pretty_names = ["R. ankle", "R. knee", "R. hip",
                     "L. hip", "L. knee", "L. ankle",
                     "Pelvis",
                     "Neck", "Torso", "Head", "Head Top",
                     "R. wrist", "R. elbow", "R. shoulder",
                     "L. shoulder", "L. elbow", "L. wrist",
                     ]


    @classmethod
    def indices(cls):
        return cls._indices

    @classmethod
    def flip_lr_indices(cls):
        return cls._flip_lr_indices

    @classmethod
    def bodypart_indices(cls):
        return cls._bodypart_indices

    @classmethod
    def limb_indices(cls):
        return cls._limb_indices

    @classmethod
    def names(cls):
        return cls._names

    @classmethod
    def pretty_names(cls):
        return cls._pretty_names

    @classmethod
    def joints_to_bodyparts(cls, joint_annotation):
        has_visibility_flag = (joint_annotation.shape[1] == 3)
        if has_visibility_flag:
            joint_dim = 3
        else:
            joint_dim = 2
        bodyparts = np.empty((cls.num_bodyparts, 2, joint_dim), dtype=np.float32)
        for i, indices in enumerate(cls.bodypart_indices()):
            bodyparts[i] = joint_annotation[indices]
        return bodyparts

    def __init__(self):
        pass


class H36MOrder17POriginalOrder:
    """
    Helper for the Human3.6M 17-point pose definition in its original order
    i.e. simply filtering the 17 relevant points from the original 32-point order
    """

    pelvis = 0
    r_hip = 1
    r_knee = 2
    r_ankle = 3
    l_hip = 4
    l_knee = 5
    l_ankle = 6
    torso = 7
    neck = 8
    head = 9
    head_top = 10
    l_shoulder = 11
    l_elbow = 12
    l_wrist = 13
    r_shoulder = 14
    r_elbow = 15
    r_wrist = 16

    num_points = 17

    _indices = [pelvis,
                r_hip, r_knee, r_ankle,
                l_hip, l_knee, l_ankle,
                torso, neck, head, head_top,
                l_shoulder, l_elbow, l_wrist,
                r_shoulder, r_elbow, r_wrist,
                ]

    @classmethod
    def indices(cls):
        return cls._indices


    @classmethod
    def to_our_17p_order(cls):
        """
        Matches the official 17point order
        :return:
        """
        return [cls.r_ankle, cls.r_knee, cls.r_hip,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.pelvis,
                cls.neck, cls.torso,
                cls.head, cls.head_top,
                cls.r_wrist, cls.r_elbow, cls.r_shoulder,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist,
                ]

    def __init__(self):
        pass

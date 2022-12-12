# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import os
import re

import numpy as np
import copy

from common.dataset.skeleton import Skeleton
from common.dataset.mocap_dataset import MocapDataset
from common.dataset.h36m_dataset import h36m_skeleton, Human36mDataset


# Keypoint reordering from AMASS (and the custom H36m joint regressor) to our 17-point order
# Note that the regressor for AMASS produces a slightly different 17-point order compared to the "official" order
amass_reorder = [6, 5, 4,  # right leg
                 1, 2, 3,  # left leg]
                 0,  # root
                 8, 7,  # neck, thorax
                 9, 10,  # head
                 16, 15, 14,  # right arm
                 11, 12, 13,  # left arm
                 ]

# Skeleton for the 17point version, in custom order (similar to MPII Human Pose order)
h36m_skeleton = Skeleton(parents=[1, 2, 6, 6, 3, 4, -1, 8, 6, 7, 9, 12, 13, 7, 7, 14, 15],
                         joints_left=[3, 4, 5, 14, 15, 16],
                         joints_right=[0, 1, 2, 11, 12, 13])

# Each element specifies: (Dataset, Subject, Action)
# Each entry is interpreted as a regex
amass_splits = {
    "train":       [("CMU", ".*", ".*"),
                    ("DanceDB", ".*", ".*"),
                    ("MPILimits", ".*", ".*"),
                    ("TotalCapture", ".*", ".*"),
                    ("EyesJapanDataset", ".*", ".*"),
                    ("HUMAN4D", ".*", ".*"),
                    ("KIT", ".*", ".*"),
                    ("BMLhandball", ".*", ".*"),
                    ("BMLmovi", ".*", ".*"),
                    ("BMLrub", ".*", ".*"),
                    ("EKUT", ".*", ".*"),
                    ("TCDhandMocap", ".*", ".*"),
                    ("ACCAD", ".*", ".*"),
                    ("Transitionsmocap", ".*", ".*"),
                    ],
    "val":         [("MPIHDM05", ".*", ".*"),
                    ("SFU", ".*", ".*"),
                    ("MPImosh", ".*", ".*"),
                    ],

    "train_debug": [("CMU", ".*", ".*"),
                    ],
    "val_debug":   [("CMU", ".*", ".*"),
                    ],
}


class AMASSDataset(MocapDataset):

    def __init__(self, path, h36m_path, split, downsample=1, h36m_cameras=None):
        super().__init__(fps=50, skeleton=h36m_skeleton)

        if h36m_cameras is None:
            cam_dataset = Human36mDataset(h36m_path)
            self._cameras = copy.deepcopy(cam_dataset.cameras())
            del (cam_dataset)
        else:
            self._cameras = copy.deepcopy(h36m_cameras)

        self.split = split
        dataset_filter = amass_splits[split] if type(self.split) is str else self.split

        datasets = [d for d in sorted(os.listdir(path)) if os.path.splitext(d)[1] == ".npz"]

        self._data = {}
        for dataset_file in datasets:
            dataset = os.path.splitext(dataset_file)[0]
            dataset_matches = [pattern for pattern in dataset_filter if re.fullmatch(pattern[0], dataset) is not None]
            if len(dataset_matches) > 0:
                print(dataset)
                # Load serialized dataset
                data = np.load(os.path.join(path, dataset_file), allow_pickle=True)['positions_3d'].item()
                self._data[dataset] = {}

                for subject, actions in data.items():
                    subject_matches = [pattern for pattern in dataset_matches if
                                       re.fullmatch(pattern[1], subject) is not None]
                    if len(subject_matches) > 0:
                        self._data[dataset][subject] = {}
                        for action_name, sequence_dict in actions.items():
                            action_matches = [pattern for pattern in subject_matches if
                                              re.fullmatch(pattern[2], action_name) is not None]
                            if len(action_matches) > 0:
                                assert sequence_dict["frame_rate"] == 50.
                                positions = sequence_dict["positions_3d"].astype(np.float32)
                                frame_rate = int(sequence_dict["frame_rate"])

                                # Reorder to our 17-point order
                                positions = positions[:, amass_reorder]

                                if downsample > 1:
                                    positions = positions[::downsample]
                                self._data[dataset][subject][action_name] = {
                                    'dataset': dataset,
                                    'subject': subject,
                                    'action': action_name,
                                    'positions': positions.copy(),
                                    'frame_rate': frame_rate,
                                }

    def supports_semi_supervised(self):
        return False

# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import os


all_subjects = [
    'S1',
    'S5',
    'S6',
    'S7',
    'S8',
    'S9',
    'S11',
]

subjects_by_split = {
    'trainval': [
        'S1',
        'S5',
        'S6',
        'S7',
        'S8',
    ],
    'test':     [
        'S9',
        'S11',
    ],
    'train':    [
        'S1',
        'S5',
        'S6',
        'S7',
    ],
    'val':      [
        'S8',
    ],
    'S8':      [
            'S8',
        ],
    'S9':      [
            'S9',
        ],
    'S11':      [
            'S11',
        ],
}

actions = [
    'Directions',
    'Discussion',
    'Eating',
    'Greeting',
    'Phoning',
    'Posing',
    'Purchases',
    'Sitting',
    'SittingDown',
    'Smoking',
    'TakingPhoto',
    'Waiting',
    'Walking',
    'WalkingDog',
    'WalkTogether',
]

renamed_actions = ["Directions", "Discussion", "Eating", "Greeting",
           "Phoning", "Photo", "Posing", "Purchases",
           "Sitting", "SittingDown", "Smoking", "Waiting",
           "WalkDog", "Walking", "WalkTogether"]

cameras = [
    '54138969',
    '55011271',
    '58860488',
    '60457274',
]

resolution_per_subject = {
    'S1':  (1002, 1000),
    'S5':  (1002, 1000),
    'S6':  (1002, 1000),
    'S7':  (1002, 1000),
    'S8':  (1002, 1000),
    'S9':  (1002, 1000),
    'S11': (1002, 1000),
}

def create_image_paths(base_path, subject, action, cam_id, frame_nums):
    path_template = os.path.join(base_path,
                 "frames",
                 subject,
                 f"{action}.{cam_id}")

    return [os.path.join(path_template, f"img_{k:06d}.jpg") for k in frame_nums]

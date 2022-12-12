# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import numpy as np


class MetricHistory:
    """
    A simple utility to keep track of multiple metrics throughout a training.
    Can be used to track the best epoch for each metric.
    """

    def __init__(self):
        self.metrics = []
        self.higher = []
        self.history = dict()

    def add_metric(self, metric, higher_is_better=True):
        assert metric not in self.metrics
        self.metrics.append(metric)
        self.higher.append(higher_is_better)
        self.history[metric] = list()

    def add_data(self, metric, value, step):
        self.history[metric].append((step, value))

    def best_value(self, metric):
        hist = np.array(self.history[metric])
        m_i = self.metrics.index(metric)
        if hist.shape[0] == 0:
            return None, None
        if self.higher[m_i]:
            best_loc = np.argmax(hist[:, 1])
        else:
            best_loc = np.argmin(hist[:, 1])
        return tuple(reversed(self.history[metric][best_loc]))

    def value_at_step(self, metric, step):
        hist = np.array(self.history[metric])
        if hist.shape[0] == 0:
            return None
        steps, vals = zip(*hist)
        if step in steps:
            return vals[steps.index(step)]
        else:
            return None

    def latest_value(self, metric):
        hist = np.array(self.history[metric])
        if hist.shape[0] == 0:
            return None
        steps, vals = zip(*hist)
        return vals[np.argmax(steps)]

    def print_best(self):
        for metric in self.metrics:
            value, step = self.best_value(metric)
            if "loss" in metric:
                print(f"{metric}: {value} (step {step})")
            else:
                print(f"{metric}: {value:.3f} (step {step})")

    def print_all_for_best_metric(self, metric):
        _, target_step = self.best_value(metric=metric)
        for print_metric in self.metrics:
            value = self.value_at_step(metric=print_metric, step=target_step)
            if "loss" in print_metric:
                print(f"{print_metric}: {value} (step {target_step})")
            else:
                print(f"{print_metric}: {value:.3f} (step {target_step})")

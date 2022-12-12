# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

from tensorflow import keras
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export


def scheduler_by_name(name):
    """
    Maps schedule names to actual LR schedulers.
    :param name: scheduel name.
    :return: keras-compatible schedule.
    """
    if name == "PiecewiseConstantDecay":
        return keras.optimizers.schedules.PiecewiseConstantDecay
    elif name == "CosineDecayRestarts":
        return keras.experimental.CosineDecayRestarts
    elif name == "ExponentialDecay":
        return keras.optimizers.schedules.ExponentialDecay
    elif name == "ExponentialDecayWithSteps":
        return ExponentialDecayWithSteps
    else:
        raise NotImplementedError(name)


@keras_export("keras.optimizers.schedules.ExponentialDecayWithSteps")
class ExponentialDecayWithSteps(LearningRateSchedule):
    """
    """

    def __init__(
            self,
            initial_learning_rate,
            decay_steps,
            decay_rate,
            large_decay_steps,
            large_decay_rate,
            name=None):
        """Applies exponential decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
          decay_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The decay rate.
          large_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
          large_decay_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The decay rate.
          name: String.  Optional name of the operation.  Defaults to
            'ExponentialDecay'.
        """
        super(ExponentialDecayWithSteps, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.large_decay_steps = large_decay_steps
        self.large_decay_rate = large_decay_rate
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "ExponentialDecayWithSteps") as name:
            initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)
            decay_rate = math_ops.cast(self.decay_rate, dtype)

            large_decay_steps = math_ops.cast(self.large_decay_steps, dtype)
            large_decay_rate = math_ops.cast(self.large_decay_rate, dtype)

            global_step_recomp = math_ops.cast(step, dtype)

            p = global_step_recomp / decay_steps
            p = math_ops.floor(p)

            large_p = global_step_recomp / large_decay_steps
            large_p = math_ops.floor(large_p)

            p = p - large_p

            decayed = math_ops.multiply(
            initial_learning_rate, math_ops.pow(decay_rate, p))

            large_decayed = math_ops.multiply(
            decayed, math_ops.pow(large_decay_rate, large_p), name=name)

        return large_decayed


def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps":           self.decay_steps,
        "decay_rate":            self.decay_rate,
        "large_decay_steps":     self.large_decay_steps,
        "large_decay_rate":      self.large_decay_rate,
        "name":                  self.name
    }

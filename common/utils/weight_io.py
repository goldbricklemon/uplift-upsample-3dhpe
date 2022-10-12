# Adapted from original tensorflow code: https://www.tensorflow.org/api_docs/python/tf/keras
# -*- coding: utf-8 -*-
"""
Created on 24 Jul 2021, 14:44

@author: waterplant365
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Iterable
import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.training import _is_hdf5_filepath
from tensorflow.python.keras.saving.hdf5_format import load_attributes_from_hdf5_group, _legacy_weights, \
    preprocess_weights_for_loading
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.platform import tf_logging as logging
import h5py


def resolve_weight_selector(weight_path, target_enxtension=".h5"):
    """
    Utililiy to select a weight file based on a given prefix.
    If the weight path is a complete path, it is simply returned.
    If the weight path is a prefix (e.g. "some/path/best_weights"),
    it will be resolved to the first matching file (e.g. to "some/path/best_weights_090.h5")
    :param weight_path: Path to weight file or prefix of a weight file.
    :param target_enxtension: The target extension to resolve to (e.g. ".h5")
    :return: The resolved path to the weight file.
    """
    if weight_path is None:
        return None
    _, ext = os.path.splitext(weight_path)
    if len(ext) != 0:
        return weight_path
    else:
        weight_dir, selector = os.path.split(weight_path)
        dir_content = os.listdir(weight_dir)
        weight_candidates = [s for s in dir_content if s.startswith(selector) and s.endswith(target_enxtension)]
        if len(weight_candidates) == 0:
            raise FileNotFoundError(f"Found no weights that match: {weight_path} and extension {target_enxtension}")
        else:
            weight_file = list(sorted(weight_candidates))[0]
            return os.path.join(weight_dir, weight_file)


class KerasWeightLoadingCallback:
    """
    Callback to adjust or filter weights during keras .h5 weight loading
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def __call__(self,
                 target_weight: tf.Variable,
                 weight_name: str,
                 weight_value: np.ndarray):
        """
        Transform weight values from .h5 file befor assignment to target weights.
        :param target_weight: target weights
        :param weight_name: weight name from .h5 file
        :param weight_value: weight value from .h5 file
        :return: (bool, Adjusted weights): Wether the weights were adjusted and the adjusted weights
        """
        return False, weight_value


def load_weights_with_callback(model,
                               filepath,
                               skip_mismatch=False,
                               callbacks=[],
                               verbose=True):
    """Loads all layer weights from an HDF5 weight file.

    Arguments:
        model: keras Model
        filepath: String, path to the weights file to load. For weight files in
            TensorFlow format, this is the file prefix (the same as was passed
            to `save_weights`).
        skip_mismatch: Boolean, whether to skip loading of layers where there is
            a mismatch in the number of weights, or a mismatch in the shape of
            the weight (only valid when `by_name=True`).
        callbacks: weight loading callbacks
        verbose:

    Raises:
        ImportError: If h5py is not available and the weight file is in HDF5
            format.
        ValueError: If `skip_mismatch` is set to `True` when `by_name` is
          `False`.
    """
    if dist_utils.is_tpu_strategy(model._distribution_strategy):
        if (model._distribution_strategy.extended.steps_per_run > 1 and
                (not _is_hdf5_filepath(filepath))):
            raise ValueError('Load weights is not yet supported with TPUStrategy '
                             'with steps_per_run greater than 1.')

    filepath = path_to_string(filepath)
    assert _is_hdf5_filepath(filepath)

    if h5py is None:
        raise ImportError(
            '`load_weights` requires h5py when loading weights from HDF5.')
    if not model._is_graph_network and not model.built:
        raise ValueError(
            'Unable to load weights saved in HDF5 format into a subclassed '
            'Model which has not created its variables yet. Call the Model '
            'first, then load the weights.')
    model._assert_weights_created()
    with h5py.File(filepath, 'r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        load_weights_from_hdf5_group_by_name_with_callbacks(
            f, model.layers, skip_mismatch=skip_mismatch, callbacks=callbacks, verbose=verbose)


def load_weights_from_hdf5_group_by_name_with_callbacks(
        f, layers, skip_mismatch=False, callbacks=Iterable[KerasWeightLoadingCallback], verbose=True):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    Arguments:
        f: A pointer to a HDF5 group.
        layers: a list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.
        callbacks:

    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file and skip_match=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    layers_consumed = {name: False for name in layer_names}

    # Reverse index of layer name to list of layers with name.
    index = {}
    layers_assigned = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)
            layers_assigned[layer.name] = False

    weights_consumed = {}
    weights_assigned = {}

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, layer_name in enumerate(layer_names):
        g = f[layer_name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        for layer in index.get(layer_name, []):
            if layer_name in index.keys():
                layers_assigned[layer_name] = True
                layers_consumed[layer_name] = True

            symbolic_weights = _legacy_weights(layer)
            weight_values = preprocess_weights_for_loading(
                layer, weight_values, original_keras_version, original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    logging.warning('Skipping loading of weights for '
                                    'layer {}'.format(layer.name) + ' due to mismatch '
                                                                    'in number of weights ({} vs {}).'.format(
                        len(symbolic_weights), len(weight_values)))
                    continue
                raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                                 '") expects ' + str(len(symbolic_weights)) +
                                 ' weight(s), but the saved weights' + ' have ' +
                                 str(len(weight_values)) + ' element(s).')

            weights_consumed.update({f"{layer_name}: {name}": value.shape for name, value in zip(weight_names, weight_values)})
            weights_assigned.update({f"{layer_name}: {weight.name}": weight.shape for weight in symbolic_weights})

            # Set values.
            for i in range(len(weight_values)):

                # Ask callbacks to transform weights before assignment
                adjusted = False
                for callback in callbacks:
                    did_adjust, adjusted_weights = callback(target_weight=symbolic_weights[i],
                                                            weight_name=weight_names[i],
                                                            weight_value=weight_values[i])
                    if adjusted and did_adjust:
                        raise AssertionError("Two (or more) callbacks tried to transform the same weights. "
                                             "This is not allowed")
                    adjusted = adjusted or did_adjust
                    if did_adjust:
                        weight_values[i] = adjusted_weights

                if weight_values[i] is None:
                    # Check whether callback has removed the weights
                    continue
                elif K.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                    if skip_mismatch:
                        logging.warning('Skipping loading of weights for '
                                        'layer {}'.format(layer.name) + ' due to '
                                                                        'mismatch in shape ({} vs {}).'.format(
                            symbolic_weights[i].shape,
                            weight_values[i].shape))
                        continue
                    raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                                     '"), weight ' + str(symbolic_weights[i]) +
                                     ' has shape {}'.format(K.int_shape(
                                         symbolic_weights[i])) +
                                     ', but the saved weight has shape ' +
                                     str(weight_values[i].shape) + '.')

                else:
                    weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
                    weights_consumed[f"{layer_name}: {weight_names[i]}"] = None
                    weights_assigned[f"{layer_name}: {symbolic_weights[i].name}"] = None

    K.batch_set_value(weight_value_tuples)
    if verbose:
        unconsumed_layers = [name for name, consumed in layers_consumed.items() if consumed is False]
        if len(unconsumed_layers) > 0:
            print("The following layers were not consumed from .h5 file:")
            for name in unconsumed_layers:
                print("- " + name)

        unassinged_layers = [name for name, assigned in layers_assigned.items() if assigned is False]
        if len(unassinged_layers) > 0:
            print("The following layers were not assigned any weights:")
            for name in unassinged_layers:
                print("- " + name)

        unconsumed_weights = [(name, shape) for name, shape in weights_consumed.items() if shape is not None]
        if len(unconsumed_weights) > 0:
            print("The following weights were not consumed from .h5 file:")
            for name, shape in unconsumed_weights:
                print("- " + name, shape)

        unassinged_weights = [(name, shape) for name, shape in weights_assigned.items() if shape is not None]
        if len(unassinged_weights) > 0:
            print("The following weights were not assigned any values:")
            for name, shape in unassinged_weights:
                print("- " + name, shape)

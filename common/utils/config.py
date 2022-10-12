# -*- coding: utf-8 -*-
"""
Created on 17 May 2021, 13:50

@author: waterplant365
"""


import os
import inspect
import json
import numpy as np
import warnings
import copy


# Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    def __init__(self, config_file=None, file_mode=None):
        """Init values from config file"""

        if config_file is not None:
            self.load(config_file, file_mode)

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def copy(self):
        new_config = self.__class__()
        attr = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attr = [a for a in attr if not (a[0].startswith('__') and a[0].endswith('__'))]
        for key, value in attr:
            new_config.__setattr__(key, copy.deepcopy(value))
        return new_config

    def load(self, config_file, file_mode=None):
        """
        Load config parameters from a file.
        Can be of two different formats:
        1. JSON format:
        2. Simpler text format:
           CONFIG_KEY <value-in-json-format>
           ...
        :param config_file: Path to config file.
        :param file_mode: format mode, "json" or "txt".
        Is automatically inferred from the filename extension, if None.
        :return:
        """
        assert os.path.exists(config_file)
        if file_mode is None:
            ext = os.path.splitext(config_file)[1]
            assert ext in [".txt", ".json"]
            if ext == ".txt":
                file_mode = "txt"
            else:
                file_mode = "json"

        if file_mode == "txt":
            with open(config_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip("\r\n ")
                split = line.split(" ", maxsplit=1)
                if len(split) > 0 and not split[0].startswith("#"):
                    if len(split) > 1 and len(split[1]) > 0:
                        if "\'" in split[1]:
                            warnings.warn("Avoid single quotes literals in config files. Use double quotes instead")
                            split[1] = split[1].replace("\'", "\"")
                        split[1] = split[1].lstrip(" ")
                        key = split[0]
                        # None is not json conform. Convert to null
                        split[1].replace("None", "null")
                        split[1].replace("False", "false")
                        split[1].replace("True", "true")
                        value = json.loads(split[1])
                        self.__setattr__(key, value)
        else:
            with open(config_file, "r") as f:
                root_dict = json.load(f)
            for key, value in root_dict.items():
                self.__setattr__(key, value)

    def dump(self, config_file):
        """
        Dump the complete config into a JSON file
        :param config_file: Path to output JSON file.
        :return:
        """
        attr = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attr = [a for a in attr if not (a[0].startswith('__') and a[0].endswith('__'))]
        root_dict = {}
        for key, value in attr:
            if type(value) is np.ndarray:
                value = value.tolist()
            root_dict[key] = value

        with open(config_file, "w") as f:
            json.dump(root_dict, f, indent=4, sort_keys=True)

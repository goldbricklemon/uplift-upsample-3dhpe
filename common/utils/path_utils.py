# -*- coding: utf-8 -*-
"""
Created on 12 May 2021, 16:50

@author: waterplant365
"""

import os


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def expandpath(path):
    x = os.path.expanduser(path)
    x = os.path.realpath(x)
    x = os.path.abspath(x)
    return x

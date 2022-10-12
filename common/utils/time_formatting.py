# -*- coding: utf-8 -*-
"""
Created on 01 Jun 2021, 16:35

@author: waterplant365
"""


def format_time(seconds):
    """
    Utiliy to format time (copied from keras code).
    :param seconds: time duration in seconds.
    :return: formatted time string
    """
    if seconds > 3600:
        time_string = '%d:%02d:%02d' % (seconds // 3600,
                                       (seconds % 3600) // 60, seconds % 60)
    elif seconds > 60:
        time_string = '%d:%02d' % (seconds // 60, seconds % 60)
    else:
        time_string = '%ds' % seconds
    return time_string

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:43:35 2018

@author: tvhogg
"""

import numpy as np

tth_vec = np.array([0, 0.3, 1, 3, 5, 10, 20, 30, 40])
gonx_vec = np.array([0, 0.075, 0.100, 0.06, 0.14, 0.16, 0.225, 0.295, 0.415])


def get_gonx_offset(tth):
    gonx = np.interp(tth, tth_vec, gonx_vec, left=0, right=4.15)
    return gonx

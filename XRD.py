#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 05:07:10 2017

@author: scott
"""
import numpy as np
from matplotlib import pyplot as plt

def integrate_peak(x, y, xspan, background='linear', 
                   background_points=4, 
                   ax=None, color='k', fill_color='g'):
    if ax == 'new':
        fig, ax = plt.subplots()
    
    i_start = next(i for i, x_i in enumerate(x) if x_i>xspan[0])
    y_prepeak = np.mean(y[i_start-background_points:i_start])
    i_finish = next(i for i, x_i in enumerate(x) if x_i>xspan[-1])
    y_postpeak = np.mean(y[i_finish:i_finish+background_points])
    
    peak_x = np.array(x[i_start:i_finish])
    peak_y = np.array(y[i_start:i_finish])
    background = ((peak_x - xspan[0]) * y_postpeak + 
                  (xspan[-1] - peak_x) * y_prepeak) / (xspan[-1] - xspan[0])
    
    I = np.trapz(peak_y-background, peak_x)
    
    if ax is not None:
        if color is not None:
            ax.plot(peak_x, peak_y, color=color)
            ax.plot(peak_x, background, '--', color=color)
        if fill_color is not None:
            ax.fill_between(peak_x, peak_y, background, where=peak_y>background,
                            facecolor=fill_color, interpolate=True)
        return I, ax
    return I

def interp_with_zeros(x, x_sub, y_sub):
    '''
    Turns out this handy little function is covered by numpy. It is equivalent
    to np.interp(x, x_sub, y_sub, left=0, right=0)
    '''
    y = np.zeros(np.size(x))
    mask = np.logical_and(x_sub[0]<x and x<x_sub[-1])
    y[mask] = np.interp(x[mask], x_sub, y_sub)
    return y
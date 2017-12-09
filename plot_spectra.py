#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:36:10 2017

@author: scott
"""

import os
import re
import numpy as np
from matplotlib import pyplot as plt


def plot_spectrum(data, ax='new', tth_str='TwoTheta', pd_str='pd6', color='k', 
                  normalize=False, tth_normalize='max',
                  removebackground=False, background=None):
    
    tth = data[tth_str]
    pd = data[pd_str]

    if removebackground:
        if background is None: #just remove the minimum value
            b = min(pd)
            background = [b for i in range(len(pd))]
        #NOTE: background must be pre-normalized, and same length as pd
        pd = pd - background
  
    if normalize:
        if tth_normalize == 'max':
            pd_0 = max(pd)
        else:
            pd_0 = np.interp(tth_normalize, tth, pd)
        if type(normalize) not in [int, float]:
            normalize = 1
        pd = normalize * pd / pd_0

            
    if ax == 'new':
        fig, ax = plt.subplots()
    ax.plot(tth, pd, color=color)
    ax.set_xlabel('TwoTheta / [deg]')
    ax.set_ylabel('counts')
    return ax

def get_alpha_scan_datasets_from_path(path):
    
    files = get_alpha_scan_files(path)
    alphas, fs = dict_to_sorted_lists(files)
    datasets = {}
    for alpha, f in zip(alphas, fs):
        print('working on apha = ' +str(alpha))
        datasets[alpha] = []
        for fi in f:
            filepath = path + os.sep + fi
            datasets[alpha] += load_from_csv(filepath, multiset=True)
    return datasets

def plot_alpha_scan_from_path(path, ax='new', legend=True):
    if ax=='new':
        fig, ax = plt.subplots()
        
    datasets = get_alpha_scan_datasets_from_path(path)
    for alpha, datalist in datasets.items():
        N = len(datalist)
        for i, data in enumerate(datalist):
            label = 'alpha=' + str(alpha)
            if N > 1:
                label += '_' + str(i)
            try:
                ax.plot(data['TwoTheta'], data['pd6'], label=label)
            except KeyError:
                print('No plottable data here! Skipping this value')
        if legend:
            ax.legend()
    return ax

def dict_to_sorted_lists(d, sortby='keys'):
    keys = list(d.keys())
    values = list(d.values())
    if sortby == 'keys':
        I = np.argsort(keys)
    elif sortby == 'values':
        I = np.argsort(values)
    keys = [keys[i] for i in I]
    values = [values[i] for i in I]    
    return keys, values


def get_angle(filename):
    match = re.search(r'_\d+p\d+_', filename)
    if match is None:
        return None
    anglestr = match.group()[1:-1]
    angle = float(anglestr.replace('p','.'))
    return angle


def get_alpha_scan_files(path):
    lslist = os.listdir(path)
    files = {}
    for f in [f for f in lslist if f[-9:] == 'scan1.csv' 
              and 'timescan' not in f
              and 'peakshiftwithdepth' not in f
              and 'detailed' not in f]:
        angle = get_angle(f)
        if angle is None:
            print('can\'t read angle from file ' + f)
            continue
        if angle not in files:
            files[angle] = [f]
        else:
            files[angle] += [f]
    return files

def get_empty_set(cols, title=''):
        #get the colheaders and make space for data   
    data = {}
    data[title] = title
    for col in cols:
        data[col] = []
    data['data cols'] = cols
    return data


def numerize(data):
    for col in data['data cols']: #numerize!
        data[col] = np.array(data[col])    


def load_from_csv(filepath, multiset=False):
    
    f = open(filepath,'r') # read the file!
    lines = f.readlines()
    colheaders = [col.strip() for col in lines[0].split(',')]
    data = get_empty_set(colheaders, title=filepath)
    datasets = []
    
    for line in lines[1:]: #put data in lists!
        vals = [val.strip() for val in line.split(',')]
        not_data = []
        newline = {}
        for col, val in zip(colheaders, vals):
            if col in data['data cols']:
                try:
                    val = float(val)
                except ValueError:
                    print('value ' + val + ' of col ' + col + ' is not data.')
                    not_data += [col]   
            newline[col] = val
        if len(not_data) == len(data['data cols']):
            print('it looks like there is another data set appended!')
            if multiset:
                print('continuing to next set.')
                numerize(data)
                datasets += [data.copy()]
                colheaders = [val.strip() for val in vals]
                data = get_empty_set(colheaders)
                continue
            else:
                print('returning first set.')
                numerize(data)
                return data
        else:    
            for col in not_data:
                data['data cols'].remove(col)
                print('column ' + col + ' removed from \'data cols \'.')

        for col, val in zip(colheaders, vals):
            data[col] += [newline[col]]
    
    numerize(data)
    datasets += [data]
    if multiset:
        return datasets
    return data

    
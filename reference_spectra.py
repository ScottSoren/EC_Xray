#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 07:22:43 2017

@author: scott
"""

'''
This is a fill-in untill I write something to calculate powder diffraction
from an ase object with periodic boundary conditions.
Or, better, query the data from the materials project.
'''

from matplotlib import pyplot as plt

from .twotheta import get_tth

xrd_ref = {}

# entries are lists of (d, I, name) where d is spacing and I
xrd_ref['Cu'] = [(2.091, 100, '111'), 
                 (1.811, 47.6, '200'), 
                 (1.280, 26.5, '220'),
                 (1.092, 26.5, '311'),
                 (1.045, 7.21, '222'),
                 (0.905, 2.65, '400'),
                 (0.831, 6.73, '331'),
                 (0.810, 5.85, '420'),
                 (0.739, 3.49, '422'),
                 (0.697, 2.48, '511')]

xrd_ref['CuO'] = [(2.751, 6.15, '110'),
                  (2.533, 30.33, '002'),
                  (2.523, 80.57, '-111'),
                  (2.3245, 100, '111'),
                  (2.308, 23.46, '200'),
                  (1.864, 28.69, '-202'),
                  (1.713, 11.46, '020'),
                  (1.582, 15.32, '202'),
                  (1.506, 21.06, '-113'),
                  (1.419, 16.66, '022'),
                  (1.408, 16.14, '-311'),
                  (1.380, 10.21, '113'),
                  (1.376, 14.73, '220')
                  ]
xrd_ref['Cu2O'] = [(3.020, 4.15, '110'), #forbidden peak
                   (2.466, 100, '111'),
                   (2.135, 37.65, '200'),
                   (1.510, 32.48, '220'),
                   (1.288, 28.30, '311'),
                   (1.233, 6.61, '222')
                 ]

xrd_ref['Cu(OH)2'] = [(5.273, 80.60, '020'),
                      (3.712, 100, '021'),
                      (2.614, 63.38, '002'),
                      (2.486, 41.61, '111'),
                      (2.354, 15.21, '041'),
                      (2.342, 20.13, '022'),
                      (2.252, 67.33, '130'),
                      (1.712, 19.04, '150'),
                      (1.706, 44.98, '132'),
                      (1.627, 14.43, '151'),
                      (1.483, 10.78, '113'),
                      (1.432, 17.80, '152')]

xrd_ref['alpha-Ni(OH)2'] = [(8.0, 100, '001'),
                            (4.62, 100, '100'),
                            (4.0, 100, '002'),
                            (2.67, 100, '110'),
                            (2.50, 100, '111'),
                            (2.31, 100, '200'),
                            (2.0, 100, '004'),
                            (1.75, 100, '210'),
                            (1.5, 100, '301')
                            ]

xrd_ref['beta-Ni(OH)2'] = [(4.605, 100, '001'),
                           (2.707, 100, '100'),
                           (2.334, 100, '101'),
                           (1.563, 100, '110'),
                           (1.480, 100, '111'),
                           (1.354, 100, '200'),
                           (1.299, 100, '201'),
                           (1.167, 100, '202'),
                           (1.095, 100, '113')
                           ]

xrd_ref_colors = {'Cu':'brown','Cu2O':'0.5', 
                  'CuO':'k', 'Cu(OH)2':'b',
                  'alpha-Ni(OH)2':'g', 'beta-Ni(OH)2':'c'}



def plot_ref_spectrum(crystal='Cu', ax='new', height=100, color=None, 
                      label=True, 
                      lam=None, E=1.7e4):
    if ax == 'new':
        fig, ax = plt.subplots()
    if color is None:
        color = xrd_ref_colors[crystal]
        
    xrd = xrd_ref[crystal]
    
    for peak in xrd:
        d = peak[0]*1e-10
        tth = get_tth(d=d, lam=lam, E=E)
        h = peak[1] * height / 100
        label = peak[2]
        ax.plot([tth, tth], [0, h], color=color, label=label)
        if label:
            plt.text(tth+0.3, 0.9*h, label, color=color, rotation=90)
    return ax
    

if __name__ == '__main__':    
    for (d, I, name) in xrd_ref['Cu(OH)2']:
        print('(' + name + '), ' + str(I) + ' at tth = ' + str(get_tth(d=d*1e-10)))
    
    
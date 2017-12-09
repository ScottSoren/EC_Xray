# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:09:36 2017

@author: sobes
"""

import numpy as np

c = 2.997925e8    #speed of light / (m/s)
qe = 1.60219e-19  #fundamental charge / (C)
h = 6.62620e-34  #planck's constant / (J*s)
hbar = h/(2*np.pi)      #reduced planck's constant / (J*s)
NA = 6.02217e23   #Avogadro's number /(mol) or dimensionless
me = 9.10956e-31  #mass of electron / (kg)
kB = 1.38062e-23  #Boltzman constant / (J/K)

u0 = 4*np.pi*1e-7      #permeability of free space / (J*s^2/(m*C^2))
e0 = 1/(u0*c**2)     #permittivity of free space / (C^2/(J*m))

R = NA*kB            #gas constant / (J/(mol*K))                 #NA in /mol
Far = NA*qe        #Faraday's constant, C/mol
amu = 1e-3/NA       #atomic mass unit / (kg)    # amu=1g/NA     


def get_tth(hkl=[1,1,1], d=None, lam=None, a0=3.61e-10, E=1.7e4):
    if lam is None:
        lam = h * c / (E * qe)
    if d is None:
        d = a0 / (np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2))
    th = np.arcsin(lam / (2 * d)) * 180 / np.pi
    tth = 2 * th
    return tth
    
    
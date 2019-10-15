# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:19 2019

@author: Camille
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
from qutip.ui.progressbar import TextProgressBar


### Parameters
Na=15 # Truncature
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
n_a = a.dag()*a # photon number

k1 = 0 # single-photon loss rate
k2 = 1 # k2 comes from the combination of g2 and kb
alpha_inf_abs= 2 #alpha stationnary
delta = k2/100 #speed rate of the rotating pumping

##Cat states
alpha= 2
C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
C_alpha_minus = qt.coherent(Na, alpha)-qt.coherent(Na, -alpha)
C_alpha_minus = C_alpha_minus/C_alpha_minus.norm()


def coef_alpha(t):
    return(alpha_inf_abs*np.exp(1j*delta*t))

cops = [k1**0.5*a,
        k2**0.5*(a**2),[k2**0.5*Ia,coef_alpha]
        ]

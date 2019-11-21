# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:54:45 2019

@author: Camille
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.close('all')
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner
from TimeDependantCops import convert_time_dependant_cops_2

"Parameters"
Na=30 # Truncature
k2 = 1 # k2 comes from the combination of g2 and kb
k1=0 # single-photon loss rate
alpha_inf_abs= 2 #alpha stationnary
delta = alpha_inf_abs**2*k2/2 #speed rate of the rotating pumping
alpha=2 #alpha of initial state

T=np.pi /delta 
T1 =0
T2 =T1+T
T_final=T2
n_t = 1001
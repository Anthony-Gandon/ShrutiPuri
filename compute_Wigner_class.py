# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:18:11 2019

@author: Anthony Gandon
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

class compute_Wigner:
    def __init__(self, space_size, nbWigner, tempsSimul, syst):
        self.space_size = np.linspace(space_size[0],space_size[1],space_size[2])
        self.nbWigner = nbWigner
        (self.fig,self.axes) = plt.subplots(2,5,figsize=(12,8))
        self.listeWigner = []
        self.n_t = tempsSimul
        self.syst = syst # -1 for 1 cat and k to identify when there are more than k+1 cats
        self.spacing = tempsSimul//nbWigner
        
    def draw_Wigner(self,res):
        for ii in range(self.n_t):
            if ii%self.spacing==0 and ii//self.spacing<self.nbWigner:
                if (self.syst<=-1): # Only one Cat
                    wig = qt.wigner(res.states[ii], self.space_size, self.space_size, g=2)
                else: # Two cats or more
                    wig = qt.wigner(res.states[ii].ptrace(self.syst), self.space_size, self.space_size, g=2)
                self.listeWigner.append(wig)
        for ii in range(len(self.listeWigner)):
            self.axes[ii//5, ii%5].pcolor(self.space_size, self.space_size, self.listeWigner[ii], cmap='bwr', vmin=[-2/np.pi, 2/np.pi])
            self.axes[ii//5, ii%5].set_aspect('equal')

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:08:38 2019

@author: Camille
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
plt.close('all')

Na=30 #truncature
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
eps=2 #drive strength
k1=1 #photon loss term
alpha=0 #size of the coherent state (initial)

n_t=101
T_f=30 #total time of simulation
tlist=np.linspace(0,T_f,n_t)
initial_state=qt.coherent(Na,alpha)

wigner=True #if we want to print the Wigner distribution of the states

##Hamiltonian and operators
#H=-1j*(eps*a-eps.conjugate()*a.dag())
#cops=[k1**0.5*a]
#
#res=qt.mesolve(H,initial_state, tlist,cops,)

#Time dependant op
H=-1j*(eps*a-eps.conjugate()*a.dag())
T_cos=np.pi/(k1/100)#angular speed of variation of the photon loss coef
def func(t,args):
    return((np.cos(2*np.pi*t*k1/10)**2+1)**0.5)

cops=[[k1**0.5*a,func]]

res=qt.mesolve(H,initial_state, tlist,cops,)

#plot the res with Wigner
if wigner:
    xvec=np.linspace(-8, 8, 51) # wigner space size
    n_wig = 10
    spacing = n_t//n_wig
    fig, ax = plt.subplots(2,5, figsize=(12,8))
    
    print('Compute and plot wigners')
    wigs = []
    for ii in range(n_t):
        if ii%spacing==0 and ii//spacing<n_wig:
            wig = qt.wigner(res.states[ii], xvec, xvec, g=2)
            wigs.append(wig)
            
    for ii in range(len(wigs)):
        ax[ii//5, ii%5].pcolor(xvec, xvec, wigs[ii], cmap='bwr', vmin=[-2/np.pi, 2/np.pi])
        ax[ii//5, ii%5].set_aspect('equal')
        
a_t = [] # average of a versus time

for ii, t in enumerate(tlist):
    a_t.append(qt.expect(a, res.states[ii]))
    
fig, ax=plt.subplots()
ax.plot(tlist, np.real(a_t))
ax.plot(tlist, np.imag(a_t), linestyle='dashed')

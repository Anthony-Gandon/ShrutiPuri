# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:19 2019

@author: Camille
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.close('all')
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner

"Parameters"
Na=20 # Truncature
Nb = 20
k1 = 0 # single-photon loss rate
k2 = 1 # k2 comes from the combination of g2 and kb
alpha_inf= 3 #alpha stationnary
beta_inf = 3
delta = k2/10 #speed rate of the rotating pumping
alpha= 2
beta = 2

T=np.pi /delta 
tau=T/5
T1 = T/2
T2 = 3*T/2

nbWignerPlot = 15
nbCols=4

"CHOICE SLOT(0) OR SMOOTHSLOT(1)"
choice = 1

"Local Parameters"
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
n_a = a.dag()*a # photon number
Ib = qt.identity(Nb) # identity
b = qt.destroy(Nb) # lowering operator
n_b = b.dag()*b # photon number*

Iab=qt.tensor(Ia,Ib)
A = qt.tensor(a,Ib)
B = qt.tensor(Ia,b)


eps_2=alpha_inf_abs**2*k2/2 #cf Mirrahimi NJP 2014


"Catstates"
C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
C_alpha_minus = qt.coherent(Na, alpha)-qt.coherent(Na, -alpha)
C_alpha_minus = C_alpha_minus/C_alpha_minus.norm()

C_beta_plus = qt.coherent(Nb, beta)+qt.coherent(Nb, -beta)
C_beta_plus = C_beta_plus/C_beta_plus.norm()
C_beta_moins = qt.coherent(Nb, beta)-qt.coherent(Nb, -beta)
C_beta_moins = C_beta_moins/C_beta_moins.norm()


time=np.linspace(0,10,1001)


"Calculates the coefficient of the hamiltonian time-dependant terms"
def time_dependant_c_ops(t,args):
    return B**2-1/2*alpha*(A+alpha)+1/2*alpha*np.exp(2j*np.pi/T*t)*(A-alpha)


H=0*A

#H=-1j*(a**2*eps_2-a.dag()**2*np.conjugate(eps_2))
cops=[k2**0.5*(A**2-alpha**2),
      time_dependant_c_ops]

"Resolution of the equation over time with mesolve"
init_state= C_alpha_plus #initial state
n_t = 1001
T=np.pi /delta #total time of simulation
tlist = np.linspace(0, 2*T, n_t)
res = qt.mesolve(H, init_state, tlist, cops, progress_bar=TextProgressBar())

#Wigner
test_Wigner = compute_Wigner([-4, 4, 51], nbWignerPlot,nbCols, n_t,-1)
test_Wigner.draw_Wigner(res)

"Plot the evolution of fidelity over time"
fidelity_list=[]
for ii,t in enumerate(tlist):
    if t<T/2:
        fidelity_list.append(qt.fidelity(res.states[ii],C_alpha_plus))
    elif t>3*T/2:
        fidelity_list.append(qt.fidelity(res.states[ii],C_alpha_plus))
    else:
        theta=np.exp(-1j*delta*(t-T/2))
        C_alpha_plus_rot=qt.coherent(Na, alpha*theta)+qt.coherent(Na, -alpha*theta)
        C_alpha_plus_rot=C_alpha_plus_rot/C_alpha_plus_rot.norm()
        fidelity_list.append(qt.fidelity(res.states[ii],C_alpha_plus_rot))
 
fig,axs= plt.subplots()
axs.plot(tlist,fidelity_list)
print("Last fidelity")
print(fidelity_list[-1])

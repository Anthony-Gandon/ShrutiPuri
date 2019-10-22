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
k1 = 0 # single-photon loss rate
k2 = 1 # k2 comes from the combination of g2 and kb
alpha_inf_abs= 2 #alpha stationnary
delta = k2/1 #speed rate of the rotating pumping
alpha= 2

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

eps_2=alpha_inf_abs**2*k2/2 #cf Mirrahimi NJP 2014


"Catstates"
C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
C_alpha_minus = qt.coherent(Na, alpha)-qt.coherent(Na, -alpha)
C_alpha_minus = C_alpha_minus/C_alpha_minus.norm()

C_y_alpha=qt.coherent(Na,alpha)+ 1j* qt.coherent(Na, -alpha)
C_y_alpha=C_y_alpha/C_y_alpha.norm()

"2 functions for the strength of the drive"
def smooth_slot(t,t1=T1,t2=T2,tau1=tau):
    res=0
    if t<=t1-tau1/2 or t>=t2+tau1/2:
        res=0
    elif t>=t1+tau1/2 and t<=t2-tau1/2:
        res=1
    elif t>t1-tau1/2 and t<t1+tau1/2 :
        res=1./2*np.sin((t-t1)/tau1*np.pi) +1./2
    elif t>t2-tau1/2 and t<t2+tau1/2 :
        res=1./2*np.sin((t2-t)/tau1*np.pi)+1./2
    return(res)
    
def slot(t,t1=T1,t2=T2):
    res = 0
    if (t>=t1 and t<=t2):
        res = 1
    return res

def drive_choice(t):
    return (1-choice)*slot(t)+choice*smooth_slot(t)

time=np.linspace(0,10,1001)
res_list=[smooth_slot(1,9,1/2,t) for t in time]
fig, axs= plt.subplots()
axs.plot(time, res_list, '+')  

"Calculates the coefficient of the hamiltonian time-dependant terms"
def coef_eps(t,args):
    res=-1j*eps_2*alpha_inf_abs/np.abs(alpha_inf_abs)
    return res*np.exp(1j*2*delta*integrate.quad(drive_choice,0,t)[0])
    
def coef_eps_conj(t,args):
    return(np.conjugate(coef_eps(t,args)))
 

H=[[a**2,coef_eps],[a.dag()**2, coef_eps_conj]]
#H=-1j*(a**2*eps_2-a.dag()**2*np.conjugate(eps_2))
cops=[k1*a,k2**0.5*a**2]

"Resolution of the equation over time with mesolve"
init_state= C_y_alpha#initial state
n_t = 1001
T=np.pi /delta #total time of simulation
tlist = np.linspace(0, 2*T, n_t)
res = qt.mesolve(H, init_state, tlist, cops, progress_bar=TextProgressBar())

#Wigner
test_Wigner = compute_Wigner([-4, 4, 51], nbWignerPlot,nbCols, n_t,-1)
test_Wigner.draw_Wigner(res)

"Plot the evolution of fidelity over time"

fidelity_list=[]
final_state=(1j*np.pi*a.dag()*a).expm()*init_state #for rotation of speed angle T 
for ii,t in enumerate(tlist):
    if t<T1:
        fidelity_list.append(qt.fidelity(res.states[ii],init_state))
    elif t>T2:
        fidelity_list.append(qt.fidelity(res.states[ii],final_state))
    else:
        theta=delta*(t-T1)
        init_state_rot=(-1j*theta*a.dag()*a).expm()*init_state
        init_state_rot=init_state_rot/init_state_rot.norm()
        fidelity_list.append(qt.fidelity(res.states[ii],init_state_rot))
 
fig,axs= plt.subplots()
axs.plot(tlist,fidelity_list)
print("Last fidelity")
print(fidelity_list[-1])

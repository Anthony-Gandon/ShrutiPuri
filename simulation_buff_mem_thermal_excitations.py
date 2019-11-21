# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:19 2019

@author: Camille
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants 
#matplotlib.rcParams['text.usetex'] = True
import scipy.integrate as integrate
plt.close('all')
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner
from TimeDependantCops import convert_time_dependant_cops

"Parameters to tune"
alpha_inf_abs=3
alpha=alpha_inf_abs
trunc=40
#Time/rate parameters"
k2=40e3 #k2= 40kHz in Lescanne
k1=10e6 #kb=13MHz in Lescanne
#Do we need to use this ? Because it squishes very quickly the cat-qubit
k2=1
k1=k2/100
#Wigner params
nbWignerPlot = 15
nbCols=4

#By default the two cats control and target have the same parameters
#We can tune them differently if we want.

"Preparation of tensor space, operators and states"
Na=trunc # Truncature
# Operators for the control 
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
n_a = a.dag()*a # photon number

"Catstates"
C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
C_alpha_minus = qt.coherent(Na, alpha)-qt.coherent(Na, -alpha)
C_alpha_minus = C_alpha_minus/C_alpha_minus.norm()

C_y_alpha=qt.coherent(Na,alpha)+ 1j* qt.coherent(Na, -alpha)
C_y_alpha=C_y_alpha/C_y_alpha.norm()
   

# First simulation without turning 
f=1e9 #buffer 1GHz  ; memory 4GHz
w=2*np.pi*f 
temp=10e-3 #T=10mK
n_th=np.exp(-scipy.constants.hbar*w/(scipy.constants.k*temp))
init_state=C_alpha_plus
#Operators
H=0*a
L2=k2**0.5*(a**2-alpha_inf_abs**2)
L1_1=(k1*(1+n_th))**0.5*a
L1_2=(k1*n_th)**0.5*a.dag()
cops_th=[L1_1,L1_2,L2]
cops=[L2,k1**0.5*a]

#Time of simulation
T_final=100/k2
n_t=501

tlist = np.linspace(0,T_final, n_t)
res_nth = qt.mesolve(H, init_state, tlist, cops_th,progress_bar=TextProgressBar())
res = qt.mesolve(H, init_state, tlist, cops,progress_bar=TextProgressBar())


#
#path='C:/Users\Camille/Documents/GitHub/CNOT_simus/Images/CNOT_coeff_depending_on_time/Cat_size2/'
#np.save(path+'table_resCNOT_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.npy'%(size_cat,trunc,k1,prop,control_on),res_CNOT.states)

    

"Plot the evolution of fidelity over time"

distance_nth_list=[] #not fidelity because it computes square root and it creates numerics errors;
distance_witness_list=[]
for ii,t in enumerate(tlist):
    distance_nth_list.append(qt.expect(res_nth.states[ii],init_state))
    distance_witness_list.append(qt.expect(res.states[ii],init_state))
# 
#Wigner of target res (to check the rotated states does the job)
plot_Wigner= compute_Wigner([-6,6,51], nbWignerPlot, nbCols, n_t,-1)
plot_Wigner.draw_Wigner(res_nth.states,"Evolution with nth")


#Plot
fig,axs= plt.subplots()
axs.plot(tlist,distance_nth_list,'+', label='N_thermal')
axs.plot(tlist,distance_witness_list, '+', label='Without n_th')
axs.set_xlabel('Time (s)')
axs.set_ylabel('Distance')
axs.legend()
axs.set_title(r'$\alpha = $ %.0f -Trunc=%.0f //$\kappa_2=$%.1f - $\kappa_1=$%.4f // $\omega /2 \pi=$ %.0f GHz'%(alpha_inf_abs,trunc,k2,k1,f/(1e9)))

#axs.set_title(r'$\alpha = $ %.0f -Trunc=%.0f //$\kappa_2=$%.1f KHz- $\kappa_1=$%.1f MHz// $\omega /2 \pi=$ %.0f GHz'%(alpha_inf_abs,trunc,k2/(1e3),k1/(1e6),f/(1e9)))
#plt.savefig(path+'fidelity_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.png'%(size_cat,trunc,k1,prop,control_on))

#Delta 
distance_nth_list=np.array(distance_nth_list)
distance_witness_list=np.array(distance_witness_list)
delta_list=distance_nth_list-distance_witness_list
fig,axs= plt.subplots()
axs.plot(tlist,delta_list,'+')
axs.set_xlabel('Time (s)')
axs.set_ylabel('Difference between the distances')
axs.set_title(r'DELTA $\alpha = $ %.1f -Trunc=%.0f //$\kappa_2=$%.1f - $\kappa_1=$%.4f // $\omega /2 \pi=$ %.0f GHz'%(alpha_inf_abs,trunc,k2,k1,f/(1e9)))
#plt.savefig(path+'fidelity_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.png'%(size_cat,trunc,k1,prop,control_on))


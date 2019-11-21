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
from TimeDependantCops import convert_time_dependant_cops_2


#By default the two cats control and target have the same parameters
#We can tune them differently if we want.

"Parameters"
Na=10 # Truncature

k2 = 1 # k2 comes from the combination of g2 and kb
k1=0 # single-photon loss rate
alpha_inf_abs= 1.5 #alpha stationnary
delta = alpha_inf_abs**2*k2/10 #speed rate of the rotating pumping
alpha=1.5 #alpha of initial state
#delta=np.pi/2
#Times 
T=np.pi /delta 
T1 =0
T2 =T1+T
T_final=T2
n_t = 201 #number of points from 0 to T_final
#from 0 to T1 : "free" evolution (with 2 photon drive H)
#from T1 to T2 : gate NOT
#from T2 to T_final : "free" evolution (with 2 photon drive H)

nbWignerPlot = 15
nbCols=4


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

if True:
    def func(t):
        return np.exp(1j*2*delta*t)
    
    def r(t, args=0):
        return np.real(func(t))
    
    def i(t, args=0):
        return np.imag(func(t))
    
    def funcA(t, args=0):
        val = complex(1-r(t)-i(t))
        return val**0.5
    
    def funcB(t, args=0):
        val = complex(1-r(t)-i(t))
        return val**0.5
    
    def funcApB(t, args=0):
        val = complex(r(t))
        return val**0.5
    
    def funcApiB(t, args=0):
        val = complex(i(t))
        return val**0.5
    
    A = k2**0.5*a**2
    B = -k2**0.5*alpha_inf_abs**2*Ia
    
    cops = [k1**0.5*a,]
    
    cops.append([A, funcA])
    cops.append([B, funcB])
    cops.append([A+B, funcApB])
    cops.append([A+1j*B, funcApiB])
    
    init_state=C_alpha_plus#initial state
    tlist = np.linspace(0, T_final, n_t)
    res_NOT = qt.mesolve(0*a, init_state, tlist, cops)#,progress_bar=TextProgressBar())
    
    res_NOT_Wigner = compute_Wigner([-6, 6, 51], nbWignerPlot,nbCols, n_t,-1)
    res_NOT_Wigner.draw_Wigner(res_NOT.states, title='Simulations NOT')

if False :    
    "Calculates the coefficient of the hamiltonian time-dependant terms"
    def coeff_rot(t):
        return(np.exp(1j*2*delta*(t-T1)*(t>=T1 and t<=T2)))
        
    H_NOT=0*a
    #H=-1j*(a**2*eps_2-a.dag()**2*np.conjugate(eps_2))
    cops1=[k1**0.5*a]
    cops2_bis=convert_time_dependant_cops_2(k2**0.5*a**2,coeff_rot,-k2**0.5*alpha_inf_abs**2*Ia)
    cops_bis=cops1+cops2_bis
    cops_noNOT=[k1**0.5*a,k2**0.5*(a**2-alpha_inf_abs**2*Ia)]

    "Resolution of the equation over time with mesolve"
    init_state=C_alpha_plus#initial state
    tlist = np.linspace(0, T_final, n_t)
    res_NOT = qt.mesolve(H_NOT, init_state, tlist, cops, progress_bar=TextProgressBar())
    
    "Resolution of the equation without NOT" #to have the theoretical fidelity (no analytical formula)
    res_free= qt.mesolve(H_NOT, init_state, tlist, cops_noNOT,  progress_bar=TextProgressBar())
    
    
    #Wigner
    res_NOT_Wigner = compute_Wigner([-6, 6, 51], nbWignerPlot,nbCols, n_t,-1)
    res_NOT_Wigner.draw_Wigner(res_NOT.states, title='Simulations NOT')
    res_free_Wigner= compute_Wigner([-6,6, 51], nbWignerPlot,nbCols, n_t,-1)
    res_free_Wigner.draw_Wigner(res_free.states, title='Simulations without NOT')
    

  
#  
#    
#"Plot the evolution of fidelity over time"
#target_res=[] #to check the Wigner
#fidelity_NOT_list=[]
#fidelity_free_list=[]
#for ii,t in enumerate(tlist):
#    if t<=T1:
#        current_theta=0
#    elif (t>T1) and (t<T2):
#        current_theta=delta*(t-T1)
#    else :
#        current_theta=np.pi
#    state_rot= (-1j*current_theta*a.dag()*a).expm()*init_state
#    state_rot=state_rot/state_rot.norm()
#    target_res.append(state_rot)
#    fidelity_NOT_list.append(qt.fidelity(res_NOT.states[ii],state_rot))
#    fidelity_free_list.append(qt.fidelity(res_free.states[ii],init_state))
# 
##Wigner of target res (to check the rotated states does the job)
#target_Wigner= compute_Wigner([-6,6,51], nbWignerPlot, nbCols, n_t,-1)
#target_Wigner.draw_Wigner(target_res,"Rotated states")
#
#
#"Plot"
#fig,axs= plt.subplots()
##FIT ?
#kappa_fit=k1*alpha_inf_abs**2*np.exp(-4*alpha_inf_abs**2)
#exp_list=np.sqrt((1+np.exp(-kappa_fit*tlist))/2)
#exp_list=np.sqrt((1+np.exp(-k1*2*alpha_inf_abs**2*tlist))/2)
#axs.plot(tlist,fidelity_NOT_list,'+',label='wiht NOT')
#axs.plot(tlist,fidelity_free_list, '+',label='without NOT')
##axs.plot(tlist,exp_list,'+', label='fit')
##axs.text(9./10*T_final,1,str(fidelity_free_list[-1]-fidelity_NOT_list[-1]))
#axs.legend()
#print("Last fidelity")
#print(fidelity_NOT_list[-1])

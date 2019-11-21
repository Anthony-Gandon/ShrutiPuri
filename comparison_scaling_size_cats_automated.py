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


"Parameters that will stay fixed"
k2 = 1 
k1=k2/1000 # single-photon loss rate

"List to keep the results"
size_cats=[2,3,4,5]
truncature=[30,50,80,100] #we have observed visually that those truncature were ok for the respective size
max_fidelity=[]

for (ind_alpha,alpha) in enumerate(size_cats) :
    Na=truncature(ind_alpha)

"Parameters"
Na=70 # Truncature

alpha_inf_abs=4 #alpha stationnary
delta = alpha_inf_abs**2*k2/1.5 #speed rate of the rotating pumping
alpha=4 #alpha of initial state

#Times 
T=np.pi /delta 
T1 =1./2*T
T2 =T1+T
T_final=T2+1.5*T
n_t = 1001 #number of points from 0 to T_final
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


"Calculates the coefficient of the hamiltonian time-dependant terms"
def coef_eps(t,args):
    return(-1j*eps_2*np.exp(1j*2*delta*(t-T1)*(t>=T1 and t<=T2)))

def coef_eps_conj(t,args):
    return(np.conjugate(coef_eps(t,args)))
 

H_NOT=[[a**2,coef_eps],[a.dag()**2, coef_eps_conj]]
#H=-1j*(a**2*eps_2-a.dag()**2*np.conjugate(eps_2))
cops=[k1**0.5*a,k2**0.5*a**2]

"Resolution of the equation over time with mesolve"
init_state=C_y_alpha #initial state
tlist = np.linspace(0, T_final, n_t)
res_NOT = qt.mesolve(H_NOT, init_state, tlist, cops, progress_bar=TextProgressBar())

"Resolution of the equation without NOT" #to have the theoretical fidelity (no analytical formula)
H_free=[-1j*eps_2*a**2 + 1j*eps_2*a.dag()**2] #eps_2 is real
res_free= qt.mesolve(H_free, init_state, tlist, cops,  progress_bar=TextProgressBar())


#Wigner
res_NOT_Wigner = compute_Wigner([-6, 6, 51], nbWignerPlot,nbCols, n_t,-1)
res_NOT_Wigner.draw_Wigner(res_NOT.states, title='Simulations with NOT')
res_free_Wigner= compute_Wigner([-6,6, 51], nbWignerPlot,nbCols, n_t,-1)
res_free_Wigner.draw_Wigner(res_free.states, title='Simulations without NOT')

    
"Plot the evolution of fidelity over time"
target_res=[] #to check the Wigner
fidelity_NOT_list=[]
fidelity_free_list=[]
for ii,t in enumerate(tlist):
    if t<=T1:
        current_theta=0
    elif (t>T1) and (t<T2):
        current_theta=delta*(t-T1)
    else :
        current_theta=np.pi
    state_rot= (-1j*current_theta*a.dag()*a).expm()*init_state
    state_rot=state_rot/state_rot.norm()
    target_res.append(state_rot)
    fidelity_NOT_list.append(qt.fidelity(res_NOT.states[ii],state_rot))
    fidelity_free_list.append(qt.fidelity(res_free.states[ii],init_state))
 
#Wigner of target res (to check the rotated states does the job)
target_Wigner= compute_Wigner([-6,6,51], nbWignerPlot, nbCols, n_t,-1)
target_Wigner.draw_Wigner(target_res,"Rotated states")


"Look for the maximum "
ind_time_start=int(n_t*T2/T_final) #to optimize we can look after 0.5T
ind_time_max=ind_time_start+np.argmax(np.array(fidelity_NOT_list[ind_time_start:]))
max_fidelity=fidelity_NOT_list[ind_time_max]


"Plot"
fig,axs= plt.subplots()
#FIT ?
kappa_fit=k1*alpha_inf_abs**2*np.exp(-4*alpha_inf_abs**2)
exp_list=np.sqrt((1+np.exp(-kappa_fit*tlist))/2)
exp_list=np.sqrt((1+np.exp(-k1*2*alpha_inf_abs**2*tlist))/2)
axs.plot(tlist,fidelity_NOT_list,'+',label='wiht NOT')
axs.plot(tlist,fidelity_free_list, '+',label='without NOT')
axs.plot(tlist,exp_list,'+', label='fit')
axs.vlines([T1,T2,T2+0.5*T,T2+T],0,1)
axs.plot(tlist[ind_time_max],max_fidelity,'or')
#axs.text(9./10*T_final,1,str(fidelity_free_list[-1]-fidelity_NOT_list[-1]))
axs.legend()
print("Last fidelity")
print(fidelity_NOT_list[-1])

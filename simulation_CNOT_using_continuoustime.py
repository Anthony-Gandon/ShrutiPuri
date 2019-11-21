# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:19 2019

@author: Camille
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.rcParams['text.usetex'] = True
import scipy.integrate as integrate
plt.close('all')
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner
from TimeDependantCops import convert_time_dependant_cops

"Parameters to tune"
size_cat=1.5
trunc=10
prop=2.1 #3 for size 2, 2.1 for size 3, 1.6 for size 4,
#Time/rate parameters"
k2=1
k1=0
delta=size_cat**2*k2/prop

#Set the time of simulation 
T=np.pi/delta
T1 =T
T2 =T1+T
T_final=T2+T*2
n_t = 201 #number of points from T1 to T2

#Wigner params
nbWignerPlot = 15
nbCols=4

#By default the two cats control and target have the same parameters
#We can tune them differently if we want.

"Preparation of tensor space, operators and states"
Na=trunc # Truncature
Nb= trunc
# Operators for the control 
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
n_a = a.dag()*a # photon number

#Operator for the target
Ib = qt.identity(Nb) # identity
b = qt.destroy(Nb) # lowering operator
n_b = b.dag()*b # photon number

#tensor operators
I_tensor =qt.tensor(Ia,Ib)
a_tensor = qt.tensor(a,Ib)
b_tensor = qt.tensor(Ia,b)
n_a_tensor = qt.tensor(n_a,Ib)
n_b_tensor = qt.tensor(Ia,n_b)

#Parameter of the control
k2_c = k2 #2 photon dissipation control
k1_c=k1 # single-photon loss rate control
alpha_inf_abs= size_cat #alpha stationnary
alpha= size_cat #alpha of initial state


#Parameters of the target
k2_t=k2 #2 photon dissipation control
k1_t=k1#single photon loss rate control
beta_inf_abs=size_cat #beta stationnary
beta=size_cat #beta of the initial state


"Catstates"
C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
C_alpha_minus = qt.coherent(Na, alpha)-qt.coherent(Na, -alpha)
C_alpha_minus = C_alpha_minus/C_alpha_minus.norm()

C_y_alpha=qt.coherent(Na,alpha)+ 1j* qt.coherent(Na, -alpha)
C_y_alpha=C_y_alpha/C_y_alpha.norm()

C_beta_plus = qt.coherent(Nb, beta)+qt.coherent(Nb, -beta)
C_beta_plus = C_beta_plus/C_beta_plus.norm()
C_beta_minus = qt.coherent(Nb, beta)-qt.coherent(Nb, -beta)
C_beta_minus = C_beta_minus/C_beta_minus.norm()

   
if True:
    def func_coeff(t):
        return np.exp(1j*2*delta*(t-T1)*(t>=T1 and t<=T2))

    def r(t, args=0):
        return np.real(func_coeff(t))
    
    def i(t, args=0):
        return np.imag(func_coeff(t))
    
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
    
    A = k2**0.5*(b_tensor**2-1./2*alpha_inf_abs*(a_tensor+alpha_inf_abs*I_tensor))
    B = k2**0.5*(1./2*alpha_inf_abs*(a_tensor-alpha_inf_abs*I_tensor))
    
    cops = [k1**0.5*a_tensor,k1**0.5*b_tensor]
    cops.append(k2**0.5*(a_tensor**2-alpha_inf_abs**2*I_tensor)) #stabilization control cat
    
    cops.append([A, funcA])
    cops.append([B, funcB])
    cops.append([A+B, funcApB])
    cops.append([A+1j*B, funcApiB])
    
    init_state_C=qt.coherent(Na,-alpha)
    control_on=True
    init_state_T=C_beta_plus
    init_state_tensor=qt.tensor(init_state_C,init_state_T)#initial state
    
    tlist = np.linspace(0,T_final, n_t)
    res_CNOT = qt.mesolve(0*a_tensor, init_state_tensor, tlist, cops,progress_bar=TextProgressBar())#,progress_bar=TextProgressBar())
    

res_CNOT_Wigner_C = compute_Wigner([-4,4, 51], nbWignerPlot,nbCols, n_t,0)
res_CNOT_Wigner_C.draw_Wigner(res_CNOT.states, title=r'CNOT-control // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%.0f - $\kappa_1=$%.0f// $\Delta=\frac{\alpha^2 \kappa_2}{c}$, c=%.0f //Control is on : %r//'%(size_cat,trunc,k2,k1,prop,control_on))
plt.savefig(path+'wigner_CONTROL_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.png'%(size_cat,trunc,k1,prop,control_on))

res_CNOT_Wigner_T = compute_Wigner([-4,4, 51], nbWignerPlot,nbCols, n_t,1)
res_CNOT_Wigner_T.draw_Wigner(res_CNOT.states, title=r'CNOT-target // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%.0f - $\kappa_1=$%.0f// $\Delta=\frac{\alpha^2 \kappa_2}{c}$, c=%.0f //Control is on : %r// '%(size_cat,trunc,k2,k1,prop,control_on))
plt.savefig(path+'wigner_TARGET_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.png'%(size_cat,trunc,k1,prop,control_on))


#Save the table of value
path='C:/Users\Camille/Documents/GitHub/CNOT_simus/Images/CNOT_coeff_depending_on_time/Cat_size2/'
np.save(path+'table_resCNOT_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.npy'%(size_cat,trunc,k1,prop,control_on),res_CNOT.states)

    
#    #Wigner
#    res_CNOT_Wigner_C = compute_Wigner([-4,4, 51], nbWignerPlot,nbCols, n_t,0)
#    res_CNOT_Wigner_C.draw_Wigner(res2.states, title='CNOT - control qubit')
#    res_CNOT_Wigner_T = compute_Wigner([-4,4, 51], nbWignerPlot,nbCols, n_t,1)
#    res_CNOT_Wigner_T.draw_Wigner(res2.states, title='CNOT - target qubit')
#    
##    
"Plot the evolution of fidelity over time"

target_res=[] #to check the Wigner
fidelity_C_list=[]
fidelity_T_list=[]
for ii,t in enumerate(tlist):
    #preparation of the rot state
    if t<=T1:
        current_theta=0
    elif (t>T1) and (t<T2):
        current_theta=delta*(t-T1)
    else :
        current_theta=np.pi
    state_rot= (1j*current_theta*a.dag()*a).expm()*init_state_T
    state_rot=state_rot/state_rot.norm()
    target_res.append(state_rot)
    #fidelity of target
    if control_on:
        fidelity_T_list.append(qt.fidelity(res_CNOT.states[ii].ptrace(1),state_rot))
    else:
        fidelity_T_list.append(qt.fidelity(res_CNOT.states[ii].ptrace(1),init_state_T))
    #fidelity of control
    fidelity_C_list.append(qt.fidelity(res_CNOT.states[ii].ptrace(0),init_state_C))

# 
#Wigner of target res (to check the rotated states does the job)
target_Wigner= compute_Wigner([-6,6,51], nbWignerPlot, nbCols, n_t,-1)
target_Wigner.draw_Wigner(target_res,"Rotated states")

#
"Plot"
fig,axs= plt.subplots()
axs.plot(tlist,fidelity_T_list,'+',label='target')
axs.plot(tlist,fidelity_C_list,'+',label='control')
axs.legend()
axs.set_title(r'$\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%.0f - $\kappa_1=$%.0f// $\Delta=\frac{\alpha^2 \kappa_2}{c}, c=$%.0f //Control is on : %r//'%(size_cat,trunc,k2,k1,prop,control_on))
plt.savefig(path+'fidelity_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.png'%(size_cat,trunc,k1,prop,control_on))



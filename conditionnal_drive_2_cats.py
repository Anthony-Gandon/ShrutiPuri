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
from compute_Wigner_class import compute_Wigner

"Parameters"
Na=10 # Truncature
Nb= 10

nbWignerPlot = 15
nbCols=4


"Local Parameters"
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
k2_c = 1 #2 photon dissipation control
k1_c=0 # single-photon loss rate control
alpha_inf_abs= 1.5 #alpha stationnary
alpha= 1.5 #alpha of initial state


#Parameters of the target
k2_t=1 #2 photon dissipation control
k1_t=0 #single photon loss rate control
beta_inf_abs=1.5 #beta stationnary
beta=1.5#beta of the initial state



#Parameters of the rotation
delta = alpha_inf_abs**2*k2_t/10 #speed rate of the rotating pumping
T=np.pi /delta #total time of the gate
division_simus=11

#Library of states
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



"Resolution of the equation by steps "
n_time_total = 1001 #total number of steps in the simulation
T=np.pi /delta #total time of the gate
T1 =T/2#time when we put on the CNOT gate
T2 = T1+T #time when we put off the CNOT gate 
T_final= T2+T/2 #total time of the simulation

#We resolve the equation by step 
#step I : from 0 to T1 : without CNOT
#step II : from T1 to T2 : with CNOT : steps discretize
#step III :  from T2 to T_total : without CNOT

"Step I"
init_state_C=qt.coherent(Na,-alpha)
control_on=True #control_on = True if init state is in - alpha.
init_state_T=C_alpha_plus

if T1==0:
    init_state_tensor=qt.tensor(init_state_C,init_state_T)
    tlist1=[]
    res1_states=[] #just for consistency with the rest of the program
else:
    tlist1=np.linspace(0,T1,101) # Pbm : there is not the same delta time of points in the different timelist
    #Operators
    cops1=[k1_c**0.5*a_tensor, k2_c**0.5*(a_tensor**2-alpha_inf_abs**2*I_tensor),k1_t**0.5*b_tensor, k2_t**0.5*(b_tensor**2-beta_inf_abs**2*I_tensor)]
    cops1=[k1_c**0.5*a_tensor, k1_t**0.5*b_tensor, k2_t**0.5*(b_tensor**2-beta_inf_abs**2*I_tensor)]

    H1=0*a_tensor+0*b_tensor
    
    res1 = qt.mesolve(H1, qt.tensor(init_state_C,init_state_T), tlist1, cops1, progress_bar=TextProgressBar())
    init_state_tensor=res1.states[-1]
    res1_states=res1.states

    print("step 1 done")

"Step II"
t_current=T1
res2_states=[]
tlist2=[]
if T_final>T1:
    for k in range(division_simus):
        tlist_current=np.linspace(t_current, t_current+T/(division_simus-1),101)
        current_op_C=k2_c**0.5*(a_tensor**2-alpha_inf_abs**2*I_tensor)
        current_op_T=k2_t**0.5*(b_tensor**2-alpha_inf_abs**2*I_tensor)
        current_op_T=k2_t**0.5*(b_tensor**2-1./2*alpha_inf_abs*(a_tensor+alpha_inf_abs*I_tensor)+1./2*np.exp(2*1j*np.pi/T*(t_current-T1))*alpha_inf_abs*(a_tensor-alpha_inf_abs*I_tensor))
        cops2=[k1_c**0.5*a_tensor,k1_t**0.5*b_tensor, current_op_C, current_op_T]
        cops2=[k1_c**0.5*a_tensor,k1_t**0.5*b_tensor, current_op_T]

        H2=0*a_tensor+0*b_tensor
        res_current=qt.mesolve(H2, init_state_tensor, tlist_current, cops2, progress_bar=TextProgressBar())
        init_state_tensor=res_current.states[-1]
        t_current= tlist_current[-1]
        res2_states+=res_current.states[:-1] #we merge the lists
        tlist2+=list(tlist_current[:-1])#merge the lists
        print('Step done :')
        print(k)
        print('/n')
print("step 2 done")

"Step III"
if T_final<=T2:
    res3_states=[]
    tlist3=[] #for consistency of the program
else:
    init_state_tensor=res2_states[-1]
    tlist3=np.linspace(T2,T_final,101)
    cops1=[k1_c**0.5*a_tensor, k2_c**0.5*(a_tensor**2-alpha_inf_abs**2*I_tensor),k1_t**0.5*b_tensor, k2_t**0.5*(b_tensor**2-beta_inf_abs**2*I_tensor)]
    cops1=[k1_c**0.5*a_tensor, k1_t**0.5*b_tensor, k2_t**0.5*(b_tensor**2-beta_inf_abs**2*I_tensor)]
    H1=0*a_tensor+0*b_tensor
    res3 = qt.mesolve(H1, init_state_tensor, tlist3, cops1, progress_bar=TextProgressBar())
    res3_states=res3.states
    print("step 3 done")




"Compil of results"
res_states=res1_states+res2_states+res3_states
time_list=list(tlist1)+tlist2+list(tlist3)


"Wigner"
Wigner_control=compute_Wigner([-5, 5, 51], nbWignerPlot,nbCols, len(time_list),0)
Wigner_control.draw_Wigner(res_states)
test_Wigner = compute_Wigner([-5, 5, 51], nbWignerPlot,nbCols, len(time_list),1)
test_Wigner.draw_Wigner(res_states)

#"Plot the evolution of fidelity over time"
#target_res=[] #to check the Wigner
#fidelity_list=[]
#final_state=(1j*np.pi*a.dag()*a).expm()*init_state #for rotation of speed angle T 
#for ii,t in enumerate(tlist):
#    current_theta=delta*(t-T1)*(t>T1 and t<T2)
#    state_rot= (-1j*current_theta*a.dag()*a).expm()*init_state
#    state_rot=state_rot/state_rot.norm()
#    target_res.append(state_rot)
#    fidelity_list.append(qt.fidelity(res.states[ii],state_rot))
#    
# 
##Wigner of target res
#target_Wigner= compute_Wigner([-4,4,51], nbWignerPlot, nbCols, n_t,-1)
#target_Wigner.draw_Wigner(target_res)
##
#fig,axs= plt.subplots()
#exp_list=np.exp(-tlist*2*alpha_inf_abs**2*k1)*(1-np.sqrt(2)/2)+np.sqrt(2)/2
#axs.plot(tlist,fidelity_list,'+')
#axs.plot(tlist,exp_list,'+')
#axs.text(9./5*T,1,str(fidelity_list[0]-fidelity_list[-1]))
#print("Last fidelity")
#print(fidelity_list[-1])

"Evaluation of the fidelity with the desired state"
#At the end of the gate
if control_on :
    res_control_on=res_states # to keep the data of the simu
else:
    res_control_off=res_states #to keep the data of simu
    
desired_state_T=C_alpha_plus
desired_state_C=qt.coherent(Na,-alpha)
rho_C = res_control_on[-1].ptrace(0)
rho_T = res_control_on[-1].ptrace(1)
fidelity_C=qt.fidelity(desired_state_C,rho_C)
fidelity_T=qt.fidelity(desired_state_T,rho_T)

print("Fidelity Control")
print(fidelity_C)
print("Fidelity Target")
print(fidelity_T)


# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:10:20 2019

@author: berdou
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.close('all')
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner

#
#Path to save the tab and pictures
path='C:/Users/berdou/Documents/Thèse/Posters/Images-poster/Images_CNOT/Simus_CNOT_without_loss'

ratio_speed_tab=range(1,60)

for ratio in ratio_speed_tab:


    size_cat=2
    trunc=17
        #Time/rate parameters"
    k2=1
    k1=0
    delta=k2/ratio
    #delta=k2
    #Set the time of simulation 
    T=np.pi/delta
    T1 =0
    T2 =T1+T
    T_final=T2+T/4
    if ratio<10: 
        n_t = 301 #number of points from T1 to T2
    elif ratio <=20:
        n_t=401
    elif ratio<=40:
        n_t=501
    elif ratio<=50:
        nt=601
        
   
    
    "Preparation of tensor space, operators and states"
    Na=trunc # Truncature
    Nb= trunc
    # Operators for the control 
    Ia = qt.identity(Na) # identity
    a = qt.destroy(Na) # lowering operator
    n_a = a.dag()*a # photon number
    
    #Operator for the target
    Ib = qt.identity(Nb) # identity
    b = qt.destroy(Nb) # lowering operator b
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
    C_y_beta=qt.coherent(Na,alpha)+ 1j* qt.coherent(Na, -alpha)
    C_y_beta=C_y_beta/C_y_beta.norm()
    
    
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
    
    A =k2**0.5*(b_tensor**2-1./2*alpha_inf_abs*(a_tensor+alpha_inf_abs*I_tensor))
    B =k2**0.5*(1./2*alpha_inf_abs*(a_tensor-alpha_inf_abs*I_tensor))
    
    cops = [k1**0.5*a_tensor,k1**0.5*b_tensor]
    cops.append(k2**0.5*(a_tensor**2-alpha_inf_abs**2*I_tensor)) #stabilization control cat
    
    cops.append([A, funcA])
    cops.append([B, funcB])
    cops.append([A+B, funcApB])
    cops.append([A+1j*B, funcApiB])
    
    init_state_C=C_alpha_plus
    control_on=True
    init_state_T=C_alpha_plus
    init_state_tensor=qt.tensor(init_state_C,init_state_T)#initial state
    
    tlist = np.linspace(0,T_final, n_t)
    res_CNOT = qt.mesolve(0*a_tensor, init_state_tensor, tlist, cops,progress_bar=TextProgressBar())#,progress_bar=TextProgressBar())
    
    #np.save(path+'tab_values',res_CNOT.states)
    # cannot save the data because it is an array of Quantum objects
    name='density_without_loss_k2_over_%d'%(ratio)  
    qt.qsave(res_CNOT,path+name)


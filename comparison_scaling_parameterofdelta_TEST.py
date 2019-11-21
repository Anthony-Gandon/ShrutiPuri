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
k1=1/1000 # single-photon loss rate
alpha=3
alpha_inf_abs=3 #size of the init cat and the stabilization drive
Na=50 #truncature
nbWignerPlot = 15
nbCols=4

check_plots= True

"List to keep the results"
factors_prop_list=[1+k/10 for k in range(30)]
max_fidelity_list=[]
plt.close('all')
for prop in factors_prop_list :
    delta=alpha_inf_abs**2*k2/prop
    #Times 
    T=np.pi /delta 
    T1 =1./2*T
    T2 =T1+T
    T_final=T2+3*T
    n_t = 1001 #number of points from 0 to T_final
#from 0 to T1 : "free" evolution (with 2 photon drive H)
#from T1 to T2 : gate NOT
#from T2 to T_final : "free" evolution (with 2 photon drive H)

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
    init_state=qt.coherent(Na,alpha) #initial state
    tlist = np.linspace(0, T_final, n_t)
    res_NOT = qt.mesolve(H_NOT, init_state, tlist, cops, progress_bar=TextProgressBar())


##Wigner
    res_NOT_Wigner = compute_Wigner([-6, 6, 51], nbWignerPlot,nbCols, n_t,-1)
    res_NOT_Wigner.draw_Wigner(res_NOT.states, title='Simulations with NOT')
#    res_free_Wigner= compute_Wigner([-6,6, 51], nbWignerPlot,nbCols, n_t,-1)
#    res_free_Wigner.draw_Wigner(res_free.states, title='Simulations without NOT')

    
    "Plot the evolution of fidelity over time"
    target_res=[] #to check the Wigner
    fidelity_NOT_list=[]
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

  
    #Wigner of target res (to check the rotated states does the job)
    target_Wigner= compute_Wigner([-6,6,51], nbWignerPlot, nbCols, n_t,-1)
    target_Wigner.draw_Wigner(target_res,"Rotated states")


    "Look for the maximum "
    ind_time_start=int(n_t*T2/T_final) #to optimize we can look after 0.5T
    ind_time_max=ind_time_start+np.argmax(np.array(fidelity_NOT_list[ind_time_start:]))
    max_fidelity=fidelity_NOT_list[ind_time_max]
    max_fidelity_list.append(max_fidelity)
    print("Step : " +str(prop)+ ' done')
    print('\n')
    
    "Plots saved to check it is ok"
    if check_plots:
        fig,axs= plt.subplots()
        axs.plot(tlist,fidelity_NOT_list,'+')
        axs.plot(tlist[ind_time_max],max_fidelity,'or')
        path='C:/Users/Camille/Documents/GitHub/CNOT_simus/Images/Optimal_prop_factor_checkings/cat_plus/'
        name1='fidelity_opt_prop_alphais%.0f_prop'%(alpha)
        name2=str(prop)+'.png' 
        fig.savefig(path+name1+name2)
        plt.close()
        

"Plotting the results"
prop_max_ind=np.argmax(max_fidelity_list)

fig, ax= plt.subplots()
ax.plot(factors_prop_list,max_fidelity_list,'+')
ax.plot(factors_prop_list[prop_max_ind],max_fidelity_list[prop_max_ind],'+r')
ax.set_xlabel(r'Factors of proportionality for $\Delta$')
ax.set_ylabel('Fidelity max')
ax.set_title(r'$\alpha=%.0f$ ; trunc=%.0f ;$\Delta=\kappa_2*|\alpha|^2*\frac{1}{prop}$ ; $\kappa_1=\frac{\kappa_2}{1000}$ ; coherent'%(alpha,Na))
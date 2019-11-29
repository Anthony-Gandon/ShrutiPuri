# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:27:03 2019

@author: berdou
"""


import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.close('all')
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner
nbWignerPlot=15
nbCols=4

#
#Path of saved res
path='C:/Users/berdou/Documents/Thèse/Posters/Images-poster/Images_CNOT/'


### SIMUS WITHOUT LOSSES #####

ratio_tab=range(1,10)
ratio_tab=np.array(ratio_tab)
prob_tab_C_tot=[]
prob_tab_T_tot=[]
prob_tab_C_T2=[]
prob_tab_T_T2=[]
prob_tab_C_afterT2=[]
prob_tab_T_afterT2=[]
Tgate_tab=[]

for ratio in ratio_tab:
    size_cat=2
    trunc=17
        #Time/rate parameters"
    k2=1
    k1=0
    delta=k2/ratio
    #delta=k2
    #Set the time of simulation 
    T=np.pi/delta
    Tgate_tab.append(T)
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
    res_CNOT=qt.qload(path+ 'Simus_CNOT_without_lossdensity_without_loss_k2_over_%d'%(ratio))
    tlist = np.linspace(0,T_final, n_t)
    tlist=np.array(tlist)
    
    #States
    Na=trunc
    Nb=trunc
    alpha= size_cat #alpha of initial state
    beta=size_cat
    C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
    C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
    C_beta_plus = qt.coherent(Nb, beta)+qt.coherent(Nb, -beta)
    C_beta_plus = C_beta_plus/C_beta_plus.norm()
    final_state_C=C_alpha_plus
    final_state_T= C_beta_plus
    
    #fidelity
    prob_tab_C=[]
    prob_tab_T=[]
    for (ii, t) in enumerate(tlist):
        prob_tab_C.append(qt.expect(res_CNOT.states[ii].ptrace(0), final_state_C))
        prob_tab_T.append(qt.expect(res_CNOT.states[ii].ptrace(1), final_state_T))
        
    ind_T2=int(n_t*T2/T_final)
    ind_after_T2=int(n_t*T2*1.1/T_final)
    prob_tab_C_tot.append(prob_tab_C)
    prob_tab_T_tot.append(prob_tab_T)
    
    prob_tab_C_T2.append(prob_tab_C[ind_T2])
    prob_tab_T_T2.append(prob_tab_T[ind_T2])
    prob_tab_C_afterT2.append(prob_tab_C[ind_after_T2])
    prob_tab_T_afterT2.append(prob_tab_C[ind_after_T2])

prob_tab_C_T2=np.array(prob_tab_C_T2)
prob_tab_T_T2=np.array(prob_tab_T_T2)

fig, ax= plt.subplots()
ax.plot(ratio_tab, prob_tab_C_T2,'+', label='Control')
ax.plot(ratio_tab,prob_tab_T_T2, '+', label='Target')
ax.set_xlabel(r'Ratio p: $\Delta=\frac{\kappa_2}{p}$')
ax.set_ylabel('Expect value wrt final state @ T2 ')
ax.set_title(r'Expectation value at end of gate, // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%d - $\kappa_1=$%d '%(size_cat,trunc,k2,k1,))
ax.legend()

np.save(path+'Fidelity/'+'prob_C_without_losses',prob_tab_C_T2)
np.save(path+'Fidelity/'+'prob_T_without_losses',prob_tab_T_T2)


#### CHECKING LOSSES PHOTON ####
## Checking the good proba decrease  
Tf=10*np.pi/k2
n_t=501
k1=k2/1000
a = qt.destroy(Na)
cops_decrease=[k1**0.5*a, k2**0.5*(a**2-size_cat**2)]
tlist=np.linspace(0,Tf,n_t)
init_state=C_alpha_plus
res = qt.mesolve(0*a, init_state, tlist, cops_decrease,progress_bar=TextProgressBar())

#visual checking with Wigner
res_Wigner= compute_Wigner([-4,4, 101], nbWignerPlot,nbCols, n_t,-1)
res_Wigner.draw_Wigner(res.states, title='Decrease')

proba_tab=[]
for (ii,t) in enumerate(tlist):
    proba_tab.append(qt.expect(res.states[ii],init_state))

#Fitting 
T_phi=1/(2*size_cat**2*k1)
fitting_tab=1/2*(1+np.exp(-tlist/T_phi))

fig, ax =plt.subplots()
ax.plot(tlist, proba_tab,'+',label='From simu')
ax.plot(tlist,fitting_tab,'+',label='fitting')
ax.set_xlabel('Time')
ax.set_ylabel('Proba, expect value againts init state')
ax.legend()
ax.set_title(r'Checking fit : $p(t)=\frac{1}{2}(1+e^{\frac{t}{T_\varphi}})$, with $T_\varphi=\frac{1}{2\alpha^2\kappa_1}$')
#Fitting ok 


### GATHERING OF PHOTON LOSSES + INADIABTICITY ####
Tgate_tab=np.array(Tgate_tab)
k1=k2/1000
exp_losses=1/2*(1+np.exp(-Tgate_tab/T_phi))
tot_loss_fidelity_C=exp_losses*prob_tab_C_T2
tot_loss_fidelity_T=exp_losses*prob_tab_T_T2

fig, ax= plt.subplots()
ax.plot(ratio_tab, tot_loss_fidelity_C,'+', label='Control')
ax.plot(ratio_tab,tot_loss_fidelity_T, '+', label='Target')
ax.set_xlabel(r'Ratio p: $\Delta=\frac{\kappa_2}{p}$')
ax.set_ylabel('Proba with losses @ T2 ')
ax.set_title(r'Proba (expectatio value wrt final state) at end of gate, // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%d - $\kappa_1=\frac{\kappa_2}{1000}$ '%(size_cat,trunc,k2))
ax.legend()


##GLOBAL SIMUS WITH LOSSES ##
ratio_tab_2=range(1,10)
prob_tab_C_tot_both=[]
prob_tab_T_tot_both=[]
prob_tab_C_T2_both=[]
prob_tab_T_T2_both=[]
prob_tab_C_afterT2_both=[]
prob_tab_T_afterT2_both=[]
Tgate_tab_both=[]

for ratio in ratio_tab_2:
    print(ratio)
    size_cat=2
    trunc=17
        #Time/rate parameters"
    k2=1
    k1=k2/1000
    delta=k2/ratio
    #delta=k2
    #Set the time of simulation 
    T=np.pi/delta
    Tgate_tab_both.append(T)
    T1 =0
    T2 =T1+T
    T_final=T2+T/4
    path='C:/Users/berdou/Documents/Thèse/Posters/Images-poster/Images_CNOT/Simus_CNOT_with_loss/'
    res_CNOT=qt.qload(path+'density_with_loss_k2_over_%d'%(ratio))
    n_t=len(res_CNOT.times)
    tlist =res_CNOT.times
    tlist=np.array(tlist)
    T_final=tlist[-1]
    
    Na=trunc
    Nb=trunc
    alpha= size_cat #alpha of initial state
    beta=size_cat
    C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
    C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
    C_beta_plus = qt.coherent(Nb, beta)+qt.coherent(Nb, -beta)
    C_beta_plus = C_beta_plus/C_beta_plus.norm()
    final_state_C=C_alpha_plus
    final_state_T= C_beta_plus
    
    #fidelity
    prob_tab_C_both=[]
    prob_tab_T_both=[]
    for (ii, t) in enumerate(tlist):
        prob_tab_C_both.append(qt.expect(res_CNOT.states[ii].ptrace(0), final_state_C))
        prob_tab_T_both.append(qt.expect(res_CNOT.states[ii].ptrace(1), final_state_T))
        
    ind_T2=int(n_t*T2/T_final)
    ind_after_T2=int(n_t*T2*1.1/T_final)
    prob_tab_C_tot_both.append(prob_tab_C_both)
    prob_tab_T_tot_both.append(prob_tab_T_both)
    
    prob_tab_C_T2_both.append(prob_tab_C_both[ind_T2])
    prob_tab_T_T2_both.append(prob_tab_T_both[ind_T2])
    prob_tab_C_afterT2_both.append(prob_tab_C_both[ind_after_T2])
    prob_tab_T_afterT2_both.append(prob_tab_C_both[ind_after_T2])
    
fig, ax= plt.subplots()
ax.plot(ratio_tab_2, prob_tab_C_T2_both,'+', label='Control')
ax.plot(ratio_tab_2,prob_tab_T_T2_both, '+', label='Target')
ax.set_xlabel(r'Ratio p: $\Delta=\frac{\kappa_2}{p}$')
ax.set_ylabel('Expect value wrt final state @ T2')
ax.set_title(r'Expectation value at end of gate, // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%d - $\kappa_1=\frac{\kappa_2}{1000}$ '%(size_cat,trunc,k2))
ax.legend()

## Comparison
fig, ax=plt.subplots()
ax.plot(ratio_tab_2, prob_tab_C_T2_both,color='darkblue', marker='+',linestyle='None', label='Control simus')
ax.plot(ratio_tab_2,prob_tab_T_T2_both, color='darkred',marker='+',linestyle='None', label='Target simus')
ax.plot(ratio_tab, tot_loss_fidelity_C, color='deepskyblue', marker='+',linestyle='None',label='Control reconstructed')
ax.plot(ratio_tab,tot_loss_fidelity_T, color='salmon', marker='+',linestyle='None',label='Target reconstructed')
ax.set_xlabel(r'Ratio p: $\Delta=\frac{\kappa_2}{p}$')
ax.set_ylabel('Expect value wrt final state @ T2')
ax.set_title(r'Comparison of expectation value at end of gate // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%d - $\kappa_1=\frac{\kappa_2}{1000}$ '%(size_cat,trunc,k2))
ax.legend()

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


#Path to save the tab and pictures
path='C:/Users/berdou/Documents/Thèse/Posters/Images-poster/CNOT_simus_Cy/'


"Parameters to tune"
size_cat=2
trunc=20
prop=2.1 #3 for size 2, 2.1 for size 3, 1.6 for size 4,
#Time/rate parameters"
k2=1
k1=k2/100
delta=size_cat**2*k2/prop
delta=k2/3
#delta=k2
#Set the time of simulation 
T=np.pi/delta
T1 =0
T2 =T1+T
T_final=T2+T
n_t = 201 #number of points from T1 to T2

#Wigner params
nbWignerPlot = 16
nbCols=4


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


if False: #solve equation 
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
    init_state_T=C_beta_plus
    init_state_tensor=qt.tensor(init_state_C,init_state_T)#initial state
    
    tlist = np.linspace(0,T_final, n_t)
    res_CNOT = qt.mesolve(0*a_tensor, init_state_tensor, tlist, cops,progress_bar=TextProgressBar())#,progress_bar=TextProgressBar())
    
#np.save(path+'tab_values',res_CNOT.states)
# cannot save the data because it is an array of Quantum objects
    
#Saving res 
if  False : 
    res_data=[]
    for (ii,t) in enumerate(tlist):
        res_data.append(res_CNOT.states[ii].data)
    np.save(path+'tab_size2_k2isdeltaover10',res_data)


res_CNOT_Wigner_C = compute_Wigner([-4,4, 501], nbWignerPlot,nbCols, n_t,0)
res_CNOT_Wigner_C.draw_Wigner(res_CNOT.states, title=r'CNOT-control // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%.0f - $\kappa_1=$%.0f// $\Delta=\frac{\alpha^2 \kappa_2}{c}$, c=%.0f //Control is on : %r//'%(size_cat,trunc,k2,k1,prop,control_on))
plt.savefig(path+'wigner_CONTROL_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.png'%(size_cat,trunc,k1,prop,control_on))

res_CNOT_Wigner_T = compute_Wigner([-4,4, 501], nbWignerPlot,nbCols, n_t,1)
res_CNOT_Wigner_T.draw_Wigner(res_CNOT.states, title=r'CNOT-target // $\alpha, \beta=$ %.0f -Trunc=%.0f //$\kappa_2=$%.0f - $\kappa_1=$%.0f// $\Delta=\frac{\alpha^2 \kappa_2}{c}$, c=%.0f //Control is on : %r// '%(size_cat,trunc,k2,k1,prop,control_on))
plt.savefig(path+'wigner_TARGET_size%.0f_trunc%.0f_k1_%.0f_coefspeed_%.1f_control_on%r.png'%(size_cat,trunc,k1,prop,control_on))

ind_T2=int( n_t*T2/T_final)



# Draw indiv Wigner
fig, ax=plt.subplots()
ii=int(n_t*T2*1*1/T_final)
space_size=[-4,4,1001]
space_size = np.linspace(space_size[0],space_size[1],space_size[2])
wig = qt.wigner(res_CNOT.states[ii].ptrace(0), space_size,space_size, g=2)
ax.pcolor(space_size, space_size, wig, cmap='bwr', vmin=[-2/np.pi, 2/np.pi])
ax.set_aspect('equal')
#
#class compute_Wigner:
#    def __init__(self, space_size, nbWigner,nbCols, tempsSimul, syst):
#        self.space_size = np.linspace(space_size[0],space_size[1],space_size[2])
#        self.nbWigner = nbWigner
#        self.nbCols= nbCols
#        (self.fig,self.axes) = plt.subplots((nbWigner+1)//nbCols,nbCols,figsize=(12,8))
#        self.listeWigner = []
#        self.n_t = tempsSimul
#        self.syst = syst # -1 for 1 cat and k to identify when there are more than k+1 cats
#        self.spacing = tempsSimul//nbWigner
#
#        for ii in range(self.n_t):
#            if ii%self.spacing==0 and ii//self.spacing<self.nbWigner+1:
#                if (self.syst<=-1): # Only one Cat
#                    wig = qt.wigner(list_states[ii], self.space_size, self.space_size, g=2)
#                else: # Two cats or more
#                    wig = qt.wigner(list_states[ii].ptrace(self.syst), self.space_size, self.space_size, g=2)
#                self.listeWigner.append(wig)
#        for ii in range(len(self.listeWigner)):
#            self.axes[ii//self.nbCols, ii%self.nbCols].pcolor(self.space_size, self.space_size, self.listeWigner[ii], cmap='bwr', vmin=[-2/np.pi, 2/np.pi])
#            self.axes[ii//self.nbCols, ii%self.nbCols].set_aspect('equal')
#        if title is None :
#            pass
#        else :
#            self.fig.suptitle(title)
#           

#To have the last distance valuef
ind_T2=ii
final_expect_T=qt.expect(res_CNOT.states[ind_T2].ptrace(1), init_state_T) # WARNING ! In the case of a cat state only the final state is teh initial state
final_F_T=np.sqrt(final_expect_T) # ok gives the same fidelity 
fidelity_T=qt.fidelity(res_CNOT.states[ind_T2].ptrace(1), init_state_T)

print(final_expect_T)
print(final_F_T)
print(fidelity_T)

print(np.sqrt(qt.expect(res_CNOT.states[-1].ptrace(1), init_state_T)))

final_expect_C=qt.expect(res_CNOT.states[-1].ptrace(0), init_state_C)
final_F_C=np.sqrt(final_expect_C)

fidelity_tab_C=[]
fidelity_tab_T=[]
for (ii, t) in enumerate(tlist):
    fidelity_tab_C.append(np.sqrt(qt.expect(res_CNOT.states[ii].ptrace(0), init_state_C)))
    fidelity_tab_T.append(np.sqrt(qt.expect(res_CNOT.states[ii].ptrace(1), init_state_T)))

fig, ax = plt.subplots()
ax.plot(tlist,fidelity_tab_T, '+', label='target')
ax.plot(tlist, fidelity_tab_C, '+', label='control')
ax.vlines(T2, ymin=0.65 , ymax=1)
ax.legend()
ind_start=int(n_t*T2/2*1/T_final)
indT= np.argmax(fidelity_tab_T[ind_start:])
indC=np.argmax(fidelity_tab_C[ind_start:])

if False :
    np.save(path+'fidelity_C',fidelity_tab_C)
    np.save(path+'fidelity_T',fidelity_tab_T)
    fig, ax = plt.subplots()
    ax.plot(tlist,fidelity_tab_T, label='target')
    ax.plot(tlist, fidelity_tab_C, label='control')
    ax.legend()



test=res_CNOT.states[0]
test.data
np.save('test',test.data)

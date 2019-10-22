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

### Parameters
Na=20 # Truncature
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
n_a = a.dag()*a # photon number

k1 = 0 # single-photon loss rate
k2 = 1 # k2 comes from the combination of g2 and kb
alpha_inf_abs= 2 #alpha stationnary
delta = k2/10 #speed rate of the rotating pumping
eps_2=alpha_inf_abs**2*k2/2 #cf Mirrahimi NJP 2014
T=np.pi /delta 

##Cat states
alpha= 2
C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
C_alpha_minus = qt.coherent(Na, alpha)-qt.coherent(Na, -alpha)
C_alpha_minus = C_alpha_minus/C_alpha_minus.norm()


#Hamiltonian and collapses operators

#Wrong 
#H=0*a #the dynamics is in cops
#
#def coef_alpha(t,args):
#    return(alpha_inf_abs*np.exp(1j*delta*t))
#
#cops = [k1**0.5*a,
#        k2**0.5*(a**2),[-k2**0.5*Ia,coef_alpha]
#        ] # /!\ !! Not linear with jump operator !!! (cf in the dissipator)

#Correct
def smooth_slot(T1,T2,tau,t):
    res=0
    if t<=T1-tau/2 or t>=T2+tau/2:
        res=0
    elif t>=T1+tau/2 and t<=T2-tau/2:
        res=1
    elif t>T1-tau/2 and t<T1+tau/2 :
        res=1./2*np.sin((t-T1)/tau*np.pi) +1./2
    elif t>T2-tau/2 and t<T2+tau/2 :
        res=1./2*np.sin((T2-t)/tau*np.pi)+1./2
    return(res)
    
time=np.linspace(0,10,10001)
res_list=[smooth_slot(1,9,1/2,t) for t in time]
fig, axs= plt.subplots()
axs.plot(time, res_list, '+')  

tau=T/5
def coef_eps(t,args):
    res=-1j*eps_2*alpha_inf_abs/np.abs(alpha_inf_abs)
    return res*np.exp(1j*2*delta*integrate.quad(lambda x: smooth_slot(T/2,3*T/2,T/10,x),0,t)[0])
    
def coef_eps_conj(t,args):
    return(np.conjugate(coef_eps(t,args)))
 

# Slot between T/2 and 3T/2 with arbirtary angle at the begining
#def coef_eps(t,args):
#    res=-1j*eps_2*alpha_inf_abs/np.abs(alpha_inf_abs)
#    if t<T/2:
#        pass
#    elif t>3*T/2+tau/2:
#        res*=np.exp(1j*2*delta*T)
#    else:
#        res*=np.exp(1j*2*delta*(t-T/2))
#    return(res)
#    
#def coef_eps_conj(t,args):
#    return(np.conjugate(coef_eps(t,args)))
# 


#def coef_eps(t,args):
#    return(-1j*eps_2*np.exp(1j*2*delta*(t-T/2)*(t>=T/2 and t<=3*T/2)))
#
#def coef_eps_conj(t,args):
#    return(np.conjugate(coef_eps(t,args)))
# 
 
#def coef_eps(t,args):
#    return(-1j*eps_2*np.exp(1j*2*delta*t)*smooth_slot(T/2,3*T/2,T/10,t))
#
#def coef_eps_conj(t,args):
#    return(np.conjugate(coef_eps(t,args))*smooth_slot(T/2,3*T/2,T/10,t))

H=[[a**2,coef_eps],[a.dag()**2, coef_eps_conj]]
#H=-1j*(a**2*eps_2-a.dag()**2*np.conjugate(eps_2))
cops=[k1*a,k2**0.5*a**2]

# Resolution 
init_state= C_alpha_plus #initial state
n_t = 1001
T=np.pi /delta #total time of simulation
tlist = np.linspace(0, 2*T, n_t)
res = qt.mesolve(H, init_state, tlist, cops, progress_bar=TextProgressBar())
#print("solved")
#http://qutip.org/docs/latest/guide/dynamics/dynamics-time.html

#Wigner
test_Wigner = compute_Wigner([-4, 4, 51], 39, n_t,-1)
test_Wigner.draw_Wigner(res)


#To follow the state we should not put the initial state but the initial state rotated.

#fidelity
fidelity_list=[]
for ii,t in enumerate(tlist):
    if t<T/2:
        fidelity_list.append(qt.fidelity(res.states[ii],C_alpha_plus))
    elif t>3*T/2:
        fidelity_list.append(qt.fidelity(res.states[ii],C_alpha_plus))
    else:
        theta=np.exp(-1j*delta*(t-T/2))
        C_alpha_plus_rot=qt.coherent(Na, alpha*theta)+qt.coherent(Na, -alpha*theta)
        C_alpha_plus_rot=C_alpha_plus_rot/C_alpha_plus_rot.norm()
        fidelity_list.append(qt.fidelity(res.states[ii],C_alpha_plus_rot))
 
fig,axs= plt.subplots()
axs.plot(tlist,fidelity_list)
print("Last fidelity")
print(fidelity_list[-1])
#print(qt.fidelity(res.states[-1],qt.coherent(Na,-alpha)))
# fidelity_list.append(qt.fidelity(res.states[ii],qt.coherent(Na,alpha*np.exp(-1j*delta*t))))


##Plot of fidelity when delta varies
#res_array=[]
#for i in range(4):
#    delta=k2*10**(-i+1)
#    
#    def coef_eps(t,args):
#        return(-1j*eps_2*np.exp(1j*2*delta*t))
#    
#    def coef_eps_conj(t,args):
#        return(np.conjugate(coef_eps(t,args)))
#      
#    H=[[a**2,coef_eps],[a.dag()**2, coef_eps_conj]]
#    res = qt.mesolve(H, init_state, tlist, cops, progress_bar=TextProgressBar())
#    res_array.append(res)
#
##Plot
#fig, axs = plt.subplots()
#for i in range(4):
#    axs.plot(t_list,-)
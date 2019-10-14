# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:27:25 2019

@author: antho
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:18:55 2019

@author: Raphael
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner



Na=15 # Truncature
Nb=15
Ia = qt.identity(Na) # identity
Ib = qt.identity(Nb)
a = qt.destroy(Na) # lowering operator
b = qt.destroy(Nb)
n_a = a.dag()*a # photon number
n_b = b.dag()*b # photon number
parity_a = (1j*np.pi*a.dag()*a).expm() # parity




#k1 = 0.1 # single-photon loss rate
#k2 = 1 # k2 comes from the combination of g2 and kb
#kb = 50
#g2 = (kb*k2)**0.5/2
#
#alpha_inf = 2.5 # dissipator will be (k2)**0.5*(a**2-alpha_inf**2)
#eb = -alpha_inf**2*g2.conjugate() # drive strength on the buffer, computed to give the right alpha_inf
#

#Hyperparameters
K = 1
P = 4 
#Local Parameters
k1_PCC = 0
k2_PCC = 1
k2_storage = 0
beta =  2# Nb of photons in PCC
alpha = 2 # Nb of photons in Storage
beta_inf = 2
alpha_inf = 2
chi0 = k2_PCC/30
Tp = np.pi/(4*chi0*beta)



C_beta_plus = qt.coherent(Nb, beta)+qt.coherent(Nb, -beta)
C_beta_plus = C_beta_plus/C_beta_plus.norm()
C_beta_moins = qt.coherent(Nb, beta)-qt.coherent(Nb, -beta)
C_beta_moins = C_beta_moins/C_beta_moins.norm()

PCC_0 = C_beta_plus

#C_alpha_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha)
#C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
#C_ialpha_plus = qt.coherent(Na, alpha*1j)+qt.coherent(Na, -alpha*1j)
#C_ialpha_plus = C_ialpha_plus/C_ialpha_plus.norm()
#storage_0 = C_alpha_plus + 1j*C_ialpha_plus
C_alpha_plus = qt.coherent(Nb, alpha)+qt.coherent(Nb, -alpha)
C_alpha_plus = C_alpha_plus/C_alpha_plus.norm()
C_alpha_moins = qt.coherent(Nb, alpha)-qt.coherent(Nb, -alpha)
C_alpha_moins = C_alpha_moins/C_alpha_moins.norm()
storage_0 = C_alpha_moins

##
a_s_tensor = qt.tensor(a,Ib)
b_PCC_tensor = qt.tensor(Ia,b)
n_a_tensor = qt.tensor(n_a,Ib)
n_b_tensor = qt.tensor(Ia,n_b)

#parity_a = (1j*np.pi*b_PCC_tensor.dag()*b_PCC_tensor).expm() # parity


cops = [k1_PCC**0.5*b_PCC_tensor,
        ]

def chi(t):
    return (t<Tp)*chi0

H_storage = -0*K*(a_s_tensor.dag()**2*a_s_tensor**2) + 0*P*(a_s_tensor.dag()**2+a_s_tensor**2)
H_pcc = -K*(b_PCC_tensor.dag()**2*b_PCC_tensor**2) + P*(b_PCC_tensor.dag()**2+b_PCC_tensor**2)
H_inter1 = (a_s_tensor.dag()*a_s_tensor-abs(alpha)**2)*(b_PCC_tensor+b_PCC_tensor.dag()-0*beta*qt.tensor(Ia,Ib))
H_inter = H_inter1
#qt.expect(a_s_tensor.dag()*a_s_tensor,qt.tensor(storage_0,PCC_0))
#-qt.expect(a_s_tensor.dag()*a_s_tensor,qt.tensor(storage_0,PCC_0))


# if one does not want to display wigners of the time evolution and 
# only evolution of average quantities, one can put a list of operator
# in the following list 
eops = [n_a_tensor, 
        n_b_tensor,
        ]

n_t = 101
tlist = np.linspace(0, Tp, n_t)

# return the average value evolution of given operators
##print(H_pcc.shape)
#print(H_inter.shape)
#print(storage_0.shape)
#print(PCC_0.shape)
#print(qt.tensor(storage_0,PCC_0).shape)

res = qt.mesolve([chi0*H_inter+H_storage+H_pcc,], qt.tensor(storage_0,PCC_0), tlist,cops,progress_bar=TextProgressBar())
print("solved")

print("Wigner storage")
test_Wigner1 = compute_Wigner([-4, 4, 51], 10, n_t, 0)
test_Wigner1.draw_Wigner(res)
print("Wigner PCC")
test_Wigner2 = compute_Wigner([-4, 4, 51], 10, n_t, 1)
test_Wigner2.draw_Wigner(res)
proba_PCC = [] 
proba_storage = [] 
fidelity_PCC = []
fidelity_storage = []

for ii, t in enumerate(tlist):
    rho_PCC = res.states[ii].ptrace(1)
    rho_storage = res.states[ii].ptrace(0)
    proba_PCC.append((PCC_0.dag()*rho_PCC*PCC_0)[0][0][0])
    proba_storage.append((storage_0.dag()*rho_storage*storage_0)[0][0][0])
    fidelity_PCC.append(qt.fidelity(PCC_0,rho_PCC)**2)
    fidelity_storage.append(qt.fidelity(storage_0,rho_storage)**2)
print("Storage",fidelity_storage[-1])

print("PCC",fidelity_PCC[-1])

#plt.plot(tlist,fidelity_PCC)
#plt.plot(tlist,proba_PCC)
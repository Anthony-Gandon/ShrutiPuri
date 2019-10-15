# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:17:54 2019

@author: antho
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from qutip.ui.progressbar import TextProgressBar
from compute_Wigner_class import compute_Wigner
plt.close('all')

wigner = True


Na=20 # Truncature
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
n_a = a.dag()*a # photon number
parity_a = (1j*np.pi*a.dag()*a).expm() # parity

k1 = 0.01 # single-photon loss rate
k2 = 0.1 # two-photon loss
alpha = 2 # dissipator will be (k2)**0.5*(a**2-alpha**2)
T_gate = 10
n_t = 101

psi_0 = qt.coherent(Na, alpha) # coherent state with amplitude alpha_inf
psi_plus = qt.coherent(Na, alpha)+qt.coherent(Na, -alpha) 
psi_plus = psi_plus/psi_plus.norm() # even cat /!\ normalisation

def dissipative_operator(t, args):
    return k2**0.5*(a**2-(alpha*np.exp(1j*np.pi*t/T_gate))**2)

H = 0*a


# if one does not want to display wigners of the time evolution and 
# only evolution of average quantities, one can put a list of operator
# in the following list 
eops = [parity_a, 
        a,
        n_a,
        ]
c_ops = []

tlist = np.linspace(0, T_gate, n_t)
res = qt.mesolve(dissipative_operator, psi_plus, tlist, c_ops,progress_bar=TextProgressBar())

test_Wigner = compute_Wigner([-4, 4, 51], 10, n_t,-1)
test_Wigner.draw_Wigner(res)
number_t = []
for ii, t in enumerate(tlist):
    number_t.append(qt.expect(n_a, res.states[ii]))
#fig, ax = plt.subplots(3)
#plt.plot(tlist, number_t)
#ax[1].plot(tlist, na_t)
#ax[2].plot(tlist, np.real(a_t))
#ax[2].plot(tlist, np.imag(a_t), linestyle='dashed')
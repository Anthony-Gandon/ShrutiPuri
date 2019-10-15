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

wigner = True

Na=30 # Truncature
Nb=4 
Ia = qt.identity(Na) # identity
Ib = qt.identity(Nb)
a = qt.destroy(Na) # lowering operator
b = qt.destroy(Nb)
n_a = a.dag()*a # photon number
n_b = b.dag()*b # photon number
parity_a = (1j*np.pi*a.dag()*a).expm() # parity

a_tensor = qt.tensor(a,Ib)
b_tensor = qt.tensor(Ia,b)
n_a_tensor = qt.tensor(n_a,Ib)
n_b_tensor = qt.tensor(Ia,n_b)



k1 = 0.1 # single-photon loss rate
k2 = 1 # k2 comes from the combination of g2 and kb
kb = 50
g2 = (kb*k2)**0.5/2

alpha_inf = 2.5 # dissipator will be (k2)**0.5*(a**2-alpha_inf**2)
eb = -alpha_inf**2*g2.conjugate() # drive strength on the buffer, computed to give the right alpha_inf



vac = qt.basis(Na, 0) # vacuum state of the cavity
one = qt.basis(Na, 1) # 1 fock state
psi_0 = qt.coherent(Na, alpha_inf) # coherent state with amplitude alpha_inf
psi_plus = qt.coherent(Na, alpha_inf)+qt.coherent(Na, -alpha_inf) 
psi_plus = psi_plus/psi_plus.norm() # even cat /!\ normalisation

vac_tensor = qt.tensor(vac, qt.basis(Nb,0))

half_H = g2.conjugate()*qt.tensor(a**2,b.dag()) # two photon exchange
H = half_H + half_H.dag()

half_Hd = eb*qt.tensor(Ia, b.dag()) # drive on buffer
Hd = half_Hd + half_Hd.dag()


cops = [k1**0.5*a_tensor,
        kb**0.5*b_tensor,
        ]

# if one does not want to display wigners of the time evolution and 
# only evolution of average quantities, one can put a list of operator
# in the following list 
eops = [n_a_tensor, 
        n_b_tensor,
        ]

n_t = 101
tlist = np.linspace(0, 1, n_t)
if wigner:
    # return the full state evolution
    res = qt.mesolve([H, Hd], vac_tensor, tlist, cops,)
else:
    # return the average value evolution of given operators
    res = qt.mesolve([H, Hd], vac_tensor, tlist, cops, e_ops=eops)


# Display wigner at 10 equally spaced times during the dynamics
if wigner:
    xvec=np.linspace(-4, 4, 51) # wigner space size
    n_wig = 10
    spacing = n_t//n_wig
    fig, ax = plt.subplots(2,5, figsize=(12,8))
    fig_b, ax_b = plt.subplots(2,5, figsize=(12,8))

    print('Compute and plot wigners')
    wigs = []
    wigs_b = []
    for ii in range(n_t):
        if ii%spacing==0 and ii//spacing<n_wig:
            wig = qt.wigner(res.states[ii].ptrace(0), xvec, xvec, g=2) # compute partial trace
            wig_b = qt.wigner(res.states[ii].ptrace(1), xvec, xvec, g=2)
            wigs.append(wig)
            wigs_b.append(wig_b)
            
    for ii in range(len(wigs)):
        ax[ii//5, ii%5].pcolor(xvec, xvec, wigs[ii], cmap='bwr', vmin=[-2/np.pi, 2/np.pi])
        ax[ii//5, ii%5].set_aspect('equal')
        
    for ii in range(len(wigs)):
        ax_b[ii//5, ii%5].pcolor(xvec, xvec, wigs_b[ii], cmap='bwr', vmin=[-2/np.pi, 2/np.pi])
        ax_b[ii//5, ii%5].set_aspect('equal')
        # si l'elimination adiabatic fonctionne correctement, 
        # le buffer est censé rester dans le vide
        
    # Display various quantities versus time
    
    na_t = [] 
    nb_t = [] 
    for ii, t in enumerate(tlist):
        na_t.append(qt.expect(n_a_tensor, res.states[ii]))
        nb_t.append(qt.expect(n_b_tensor, res.states[ii]))
else:
    na_t = res.expect[0]
    nb_t = res.expect[1]


fig, ax = plt.subplots()
ax.plot(tlist, na_t)
ax.plot(tlist, nb_t)
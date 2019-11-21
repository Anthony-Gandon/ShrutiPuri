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
Ia = qt.identity(Na) # identity
a = qt.destroy(Na) # lowering operator
n_a = a.dag()*a # photon number
parity_a = (1j*np.pi*a.dag()*a).expm() # parity

k1 = 0.0 # single-photon loss rate
k2 = 1 # two-photon loss
alpha_inf = 2.5 # dissipator will be (k2)**0.5*(a**2-alpha_inf**2)

vac = qt.basis(Na, 0) # vacuum state of the cavity
one = qt.basis(Na, 1) # 1 fock state
psi_0 = qt.coherent(Na, alpha_inf) # coherent state with amplitude alpha_inf
psi_plus = qt.coherent(Na, alpha_inf)+qt.coherent(Na, -alpha_inf) 
psi_plus = psi_plus/psi_plus.norm() # even cat /!\ normalisation

H = 0*a # just so that H has the good type and shape

cops = [k1**0.5*a,
        k2**0.5*(a**2-alpha_inf**2),
        ]

def func(t):
    return np.exp(1j*2*np.pi*t)

def r(t, args=0):
    return np.real(func(t))

def i(t, args=0):
    return np.imag(func(t))

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

A = k2**0.5*a**2
B = -k2**0.5*alpha_inf**2*Ia

cops = [k1**0.5*a,]

cops.append([A, funcA])
cops.append([B, funcB])
cops.append([A+B, funcApB])
cops.append([A+1j*B, funcApiB])


# if one does not want to display wigners of the time evolution and 
# only evolution of average quantities, one can put a list of operator
# in the following list 
eops = [parity_a, 
        a,
        n_a,
        ]

n_t = 101
tlist = np.linspace(0, 1, n_t)
if wigner:
    # return the full state evolution
    res = qt.mesolve([H,], psi_plus, tlist, cops,)
else:
    # return the average value evolution of given operators
    res = qt.mesolve([H,], psi_plus, tlist, cops, e_ops=eops)


# Display wigner at 10 equally spaced times during the dynamics
if wigner:
    xvec=np.linspace(-4, 4, 51) # wigner space size
    n_wig = 10
    spacing = n_t//n_wig
    fig, ax = plt.subplots(2,5, figsize=(12,8))
    
    print('Compute and plot wigners')
    wigs = []
    for ii in range(n_t):
        if ii%spacing==0 and ii//spacing<n_wig:
            wig = qt.wigner(res.states[ii], xvec, xvec, g=2)
            wigs.append(wig)
            
    for ii in range(len(wigs)):
        ax[ii//5, ii%5].pcolor(xvec, xvec, wigs[ii], cmap='bwr', vmin=[-2/np.pi, 2/np.pi])
        ax[ii//5, ii%5].set_aspect('equal')
    
    # Display various quantities versus time
    
    p_t = [] # parity versus time
    a_t = [] # average of a versus time
    na_t = [] # average photon number versus time
    for ii, t in enumerate(tlist):
        p_t.append(qt.expect(parity_a, res.states[ii]))
        na_t.append(qt.expect(n_a, res.states[ii]))
        a_t.append(qt.expect(a, res.states[ii]))
else:
    p_t = res.expect[0]
    na_t = res.expect[2]
    a_t = res.expect[1].astype(complex)



fig, ax = plt.subplots(3)
ax[0].plot(tlist, p_t)
ax[1].plot(tlist, na_t)
ax[2].plot(tlist, np.real(a_t))
ax[2].plot(tlist, np.imag(a_t), linestyle='dashed')
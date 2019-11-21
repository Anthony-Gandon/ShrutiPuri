# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:09:55 2019

@author: antho
"""

import qutip as qt
import numpy as np

def convert_time_dependant_cops(a,K,b):
    '''cops is such as c_ops = a+K(t)b ; K is an analytic function of time'''
    def K_real(t):
        return np.real(K(t))
    def K_imag(t):
        return np.imag(K(t))
    
    def Coef_1(t,args):
        return 1-K_real(t)-K_imag(t)
    def Coef_2(t,args):
        return np.abs(K(t))**2-K_real(t)-K_imag(t)
    def Coef_3(t,args):
        return K_real(t)
    def Coef_4(t,args):
        return K_imag(t)
    
    c_ops_new = [[a,Coef_1],[b,Coef_2],[a+b,Coef_3],[a+1j*b, Coef_4]] #a+1jb * coef_4 ? 
    return c_ops_new


def convert_time_dependant_cops_2(A,func,B) : #from Raph
#func is a complex exponential rotating phase
#A, B are time independant operator
#returns the list of cops to implement the dissipator D[A+func(t)B]

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
    
    cops_new=[]
    cops_new.append([A, funcA])
    cops_new.append([B, funcB])
    cops_new.append([A+B, funcApB])
    cops_new.append([A+1j*B, funcApiB])
    return(cops_new)


#def func(t):
#    return np.exp(1j*2*np.pi*t)
#
#def r(t, args=0):
#    return np.real(func(t))
#
#def i(t, args=0):
#    return np.imag(func(t))
#
#def funcA(t, args=0):
#    val = complex(1-r(t)-i(t))
#    return val**0.5
#
#def funcB(t, args=0):
#    val = complex(1-r(t)-i(t))
#    return val**0.5
#
#def funcApB(t, args=0):
#    val = complex(r(t))
#    return val**0.5
#
#def funcApiB(t, args=0):
#    val = complex(i(t))
#    return val**0.5
#
#A = k2**0.5*a**2
#B = -k2**0.5*alpha_inf**2*Ia
#
#cops = [k1**0.5*a,]
#
#cops.append([A, funcA])
#cops.append([B, funcB])
#cops.append([A+B, funcApB])
#cops.append([A+1j*B, funcApiB])

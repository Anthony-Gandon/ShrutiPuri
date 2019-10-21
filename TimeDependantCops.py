# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:09:55 2019

@author: antho
"""

import qutip as qt
import numpy as np

def convert_time_dependant_cops(a,K,b):
    '''cops is such as c_ops = a+K(t)b ; K is an analytic function of time'''
    def K_real(t, args):
        return np.real(K(t))
    def K_imag(t, args):
        return np.imag(K(t))
    
    def Coef_1(t,args):
        return 1-K_real(t)-K_imag(t)
    def Coef_2(t,args):
        return np.abs(K(t))**2-K_real(t)-K_imag(t)
    def Coef_3(t,args):
        return K_real(t)
    def Coef_4(t,args):
        return K_imag(t)
    
    c_ops_new = [[a,Coef_1],[b,Coef_2],[a+b,Coef_3],[a+1j*b]]
    return c_ops_new
    
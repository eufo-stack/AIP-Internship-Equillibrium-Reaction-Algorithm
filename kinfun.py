# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:20:17 2022

@author: sheld
"""
import numpy as np

def kinfun(t, c, k, Xr, X):
    """
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    c : TYPE 2-d Numpy array with dimentions nx1
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    Xr : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    2-d numpy array

    """
    # print(Xr)
    v = np.zeros((1, k.shape[1]), dtype='float64')
    for i in range(k.shape[1]):
        # print('Exp inputs:', c, Xr[i,:])
        # print(np.exp(c, Xr[i, :], dtype='float64'))
        v[0][i] =  k[0][i]*np.prod(np.power(c, Xr[i, :], dtype='float64')) 
        
    # print(c.shape)
    conc = np.zeros((c.shape[0]))
    for i in range(c.shape[0]):
        conc[i]=np.matmul(v, X[:, i])
    # print(c)
    return conc
        
        
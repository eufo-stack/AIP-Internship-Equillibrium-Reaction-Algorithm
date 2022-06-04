# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:02:14 2022

@author: sheld
"""
import numpy as np
from numpy import matlib
import warnings
warnings.filterwarnings('error')
def NewtonRaphson(Model, beta, c_tot, c, i):
    """


    Parameters
    ----------
    Model : TYPE 
        DESCRIPTION.
    beta : TYPE np 2-d array with dtype int64
        DESCRIPTION.
    c_tot : TYPE
        DESCRIPTION.
    c : TYPE Numpy array
        DESCRIPTION.
    i : TYPE int
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ncomp = np.size(c_tot)
    nspec= np.size(beta)
    c_tot[c_tot==0] = 1e-15
    #print(c_tot)
    it=0
    while it<=99:
        
        it = it+1
        tile = np.tile(np.transpose(c), (1, nspec))
        
        c_spec=np.multiply(beta,np.prod(np.power(tile, Model), axis=0))
        try:
            c_tot_calc = np.sum(np.multiply(Model, np.tile(c_spec, (ncomp, 1))), axis=1)
            c_tot_calc = np.transpose(c_tot_calc)
        except Warning:
            print('Warning from NR')
            print(Model, beta, c_tot, c)
            raise RuntimeError
        
        d = np.subtract(c_tot, c_tot_calc)
        if np.all(np.absolute(d)< 1e-15):
            return c_spec
        c_spec[c_spec==0]=1e-15
        c_spec[np.isinf(c_spec)]=1e-15
        c_spec[np.isnan(c_spec)]=1e-15
        J_s = np.zeros((ncomp, ncomp))
        for j in range(ncomp):
            for k in range(ncomp):
                J_s[j][k] = np.sum(np.multiply(np.multiply(Model[j], Model[k]), c_spec))
                J_s[k][j] = J_s[j][k]
        # if it==1 and i==0:
        #     print("c diag:", np.diag(c[0]))
        #     print("d.T: ", d.T)
        #     print("J_s.T: ", J_s.T)
        #print(J_s)
        # print(np.linalg.det(J_s))
        try:
            delta_c = np.matmul(np.linalg.solve(J_s.T, d.T).T,np.diag(c[0]))
        except np.linalg.LinAlgError:
            #print('Adding noise!')
            noise = (np.random.normal(0, np.abs(J_s).min()/500, size=J_s.T.shape))
            # noise = (np.random.rand(J_s.T.shape[0], J_s.T.shape[1])*J_s.min()/100000-J_s.min()/)
            try:
                delta_c = np.matmul(np.linalg.solve(J_s.T+noise, d.T).T,np.diag(c[0]))
            except np.linalg.LinAlgError:
                
                return c_spec
            
        # delta_c = np.matmul(np.linalg.lstsq(J_s.T, d.T, rcond=None)[0].T,np.diag(c[0])) 
        c = c+delta_c
        while np.any(c <= 0):
            delta_c = 0.5*delta_c
            c = c-delta_c
            if np.all(np.absolute(delta_c) < 1e-15):
                break
    if it>99:
        return [1e-15]*len(c_spec)
        #print("No convergence at C_spec({0}, :)\n".format(i))
        #raise RuntimeError("No convergence at C_spec({0}, :)\n".format(i))
    return c_spec

    

   

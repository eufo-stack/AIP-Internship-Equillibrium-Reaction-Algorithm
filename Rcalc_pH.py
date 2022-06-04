#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('run', 'ModelPredictive.py')


# In[2]:


# get_ipython().run_line_magic('run', 'NewtonRaphson.py')


# In[3]:


# get_ipython().run_line_magic('run', 'ModelPredictiveTester.py')


# In[1]:


# Newton Raphson - Sheldon
import numpy as np
import warnings
warnings.filterwarnings('error')
from numpy import matlib
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


# In[40]:


# Rcalc_pH - Eudora
np.seterr(over='ignore')
def Rcalc_pH(log_beta, C_tot, pH_m, Model, par_loose, absorbing):
    nvol = C_tot.shape[0]
    ncomp = C_tot.shape[1]
    beta = np.power(10,log_beta, dtype='float64')
    c_comp_guess = np.array([[1,1]])*1e-10
    C=np.zeros((nvol, np.size(beta)))
    for i in range(nvol):
        
        try:
            C[i,:] = NewtonRaphson(Model, beta, C_tot[i, :], c_comp_guess, i)
        except RuntimeError as nr:
            raise RuntimeError from nr
        #print(C[i,:])
        c_comp_guess = np.array([C[i,:ncomp]]);
    
    # print(C)
    #print(C[:, np.nonzero(absorbing)[0][0]])
    pH_cal = -np.log10(C[:, np.nonzero(absorbing)[0][0]])
    r = pH_m - pH_cal
    return (r, C, pH_cal)


# In[41]:


# Initial Testing
log_beta = np.array([[0,0,10,16]], dtype='float64')
C_tot = np.array([[0.01,0.01],[0.02,0.02]])
pH_m = np.array([4.4998,4.5998])
Model = np.array([[1,0,1,1],[0,1,1,2]])
par_loose = [3,4,5]
absorbing = [0,1,0,0]
print(Rcalc_pH(log_beta, C_tot, pH_m, Model, par_loose, absorbing))


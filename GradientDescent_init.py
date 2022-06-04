#!/usr/bin/env python
# coding: utf-8

# In[1]:


# view model algorithm output
get_ipython().run_line_magic('run', 'ModelPredictive.py')
get_ipython().run_line_magic('run', 'NewtonRaphson.py')
out = get_ipython().run_line_magic('run', 'ModelPredictiveTester.py')
out


# In[2]:


# imports
from NewtonRaphson import NewtonRaphson 
from ModelPredictive import predictModel
import numpy as np
from Rcalc_pH import Rcalc_pH
from numpy import *
from numpy.linalg import norm
import csv
import pandas as pd


# In[3]:


# multivariant gradient descent alg, starting with one chemical model, say ['A', 'H', 'AH', 'AHH'] (can be any)

# assumed inputs of objective function (temporary)
chem_model = np.array([[1,0,1,1],  #A
                       [0,1,1,2]]) #H
i=0
ncomp=np.size(c_comp_guess)
v_0=.050
v_inc = .000005
v_added = np.arange(0,0.005, v_inc)
nvol    = np.size(v_added); 
v_tot   = v_0+v_added
C_tot = np.zeros((nvol, ncomp))
c_0     = np.array([[0.01, 0.02]]); 
c_added    = np.array([[0, -0.15]]);
for j in range(ncomp):
    C_tot[:, j]= (v_0*c_0[0,j]+v_added*c_added[0,j])/v_tot
pH_meas_real = np.array(pd.read_csv("./aip_pH.csv",header=None)[0])[:-1] #last indexing temporary

# objective function (very slightly modified testModel method from ModelPredictive.py)
def obj_func(s1, s2):
        """
        Replicates the Data_pH.m matlab file to test if the model's pH_meas is close to pH_meas_real

        Parameters
        ----------
        AH : TYPE float
            DESCRIPTION. Guess for AH beta value
        AHH : TYPE float
            DESCRIPTION. Guess for AHH beta value

        Returns
        -------
        diff : TYPE np.double
            DESCRIPTION. Difference between measured pH from algorithm and real measured pH, taken as inner product

        """
        beta = 10**np.array([[0,0,s1,s2]], dtype='float64')
        C=np.zeros((nvol, np.size(beta)))
        c_comp_guess = np.array([[1,1]])*1e-10
        for i in range(nvol):
            C[i, :]=NewtonRaphson(chem_model, beta, C_tot[i, :], c_comp_guess, i)
            c_comp_guess=np.array([C[i,:ncomp]]);
        pH_calc=-1*np.log10(C[:, 1])
        sig_R = 0.002
        pH_meas=pH_calc+sig_R*np.random.random(pH_calc.shape)
        diff = np.linalg.norm(pH_meas_real-pH_meas, ord=2)
        return diff

# partial derivative helper functions
h = 0.0001 #step
def dfdx(x, y):
    return (obj_func(x+h, y)-obj_func(x, y))/h

def dfdy(x, y):
    return (obj_func(x, y+h)-obj_func(x, y))/h

def gradf(x, y):
    return array([dfdx(x, y), dfdy(x, y)])

# multivariant gradient descent method
def grad_descent(f, gradf, init_t, alpha, max_iter):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*gradf(t[0], t[1])
        print(t)
        iter += 1
    return t


# In[4]:


# beta guess 20, 20 - output was [10.08634257, 16.93313798]
grad_descent(obj_func, gradf, array([20.0, 20.0]), 0.01, 100)


# In[8]:


# beta guess 1, 1 - output was [9.27202248, 1.58147567]
# note - when I tried 1000 iterations the second molecule stayed between 2-3
grad_descent(obj_func, gradf, array([1.0, 1.0]), 0.01, 100)


# In[ ]:





# -*- coding: utf-8 -*-
"""Main_pH_ARS.ipynb


"""

import pandas as pd
import numpy as np
from scipy import optimize
import numpy.matlib
from nglm import nglm
from NewtonRaphson import NewtonRaphson
from Rcalc_pH import Rcalc_pH

##Data_pH code
Model = np.array([[1,0,1,1, 0], 
                  [0,1,1,2, -1]]) #Species name: ['A', 'H' ,'AH' ,'AHH']
beta = 10**np.array([[0,0,10,17, -14.0744]], dtype='float64')
c_comp_guess=np.array([[1,1]])*1e-10
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
C=np.zeros((nvol, np.size(beta)))
for i in range(nvol):
    C[i, :]=NewtonRaphson(Model, beta, C_tot[i, :], c_comp_guess, i)
    c_comp_guess=np.array([C[i,:ncomp]]);
pH_calc=-1*np.log10(C[:, 1]) #How should the 2 change for general case?
np.random.seed(4)
sig_R = 0.002
pH_meas=pH_calc+sig_R*np.random.normal(0, 1, pH_calc.shape)
pH_meas = pH_meas.tolist()
# C_tot,pH_meas,v_added=Data_pH;


spec_names = [ 'A', 'H',  'HA', 'H2A', 'OH' ];
Model = np.array([[1,0,1, 1,0], [0,1,1, 2,-1]])
par_loose  = [2, 3, 4];
absorbing = [   0,     1,     0,     0,      0   ];
log_beta   = np.array([[  0,   0,   9,  16,  -10]], dtype='float64'); 
log_beta_fit,ssq,C,pH_cal,Curv = nglm(Rcalc_pH,log_beta,C_tot,pH_meas,Model,par_loose,absorbing); 	      
sig_r = np.sqrt(ssq/(np.prod(np.shape(pH_meas)) - len(log_beta_fit)));# % sigma_r	
sig_k = sig_r * np.sqrt(np.diag(np.linalg.inv(Curv)));					      #  % sigma_par
print(log_beta_fit) 
       
#plot(pH_meas,'b'),hold on;plot(pH_cal,'*b');
#xlabel('v added');ylabel('pH');

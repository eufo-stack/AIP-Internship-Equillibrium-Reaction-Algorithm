# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:11:40 2022

@author: sheld
"""

# Newton Gauss 

from NewtonRaphson import NewtonRaphson 
import numpy as np
import time

def nglm(fname,p, C_tot, y, Model, par_loose, absorbing):
    """
    

    Parameters
    ----------
    fname : TYPE function
        DESCRIPTION. function used to get residuals
    p : TYPE list
        DESCRIPTION. initial betas
    C_tot : TYPE 2-d list
        DESCRIPTION. Initial concentrations
    y : TYPE list
        DESCRIPTION. calculated pH measurements for each concentration following Newton Raphson
    Model : TYPE 2-d numpy array/list
        DESCRIPTION. Chemical model
    par_loose : TYPE 1-d list
        DESCRIPTION. List containing indexes of p that need to be fitted
    absorbing : TYPE 1-d list
        DESCRIPTION. List with 1 nonzero value which denotes the H element column

    Returns
    -------
    p : TYPE list
        DESCRIPTION. Fitted betas
    ssq : TYPE float
        DESCRIPTION. Minimized loss of diffs between meas pH and calc pH
    C : TYPE 2-d list
        DESCRIPTION. Conctrations of each element for each solution
    A : TYPE 1-d list
        DESCRIPTION. resulting calculated pH for each solution
    Curv : TYPE square n-n array, n=len(par_loose)
        DESCRIPTION. curvature matrix

    """
    ssq_old = 1e50
    mp=0
    mu=1e-4
    delta=1e-6

    it=0
    J=None
    delta_p=np.zeros((len(par_loose), 1))
    r0_old = np.zeros((C_tot.shape[0],))
    while it<50:
        print(it, p)
        try:
            r0, C, A = fname(p, C_tot, y, Model, par_loose, absorbing)
        except RuntimeError as pH_gen:
            raise RuntimeError from pH_gen
            return 
        if J is None:
            J = np.zeros((len(r0), len(par_loose)))
        ssq = r0 @ r0
        conv_crit=(ssq_old-ssq)/(ssq_old)
        if abs(conv_crit)<=mu:
            if mp==0:
                break
            else:
                mp=0
                r0_old=r0
        elif conv_crit>mu:
            mp=mp/3
            ssq_old=ssq
            r0_old=r0
            for i in range(len(par_loose)):
                p[0, par_loose[i]]=(1+delta)*p[0, par_loose[i]]
                r, _, _2 =fname(p, C_tot, y, Model, par_loose, absorbing)
                J[:, i]=(r-r0)/(delta*p[0, par_loose[i]])
                p[0, par_loose[i]]=p[0, par_loose[i]]/(1+delta)   
        elif conv_crit < -mu:
            if mp==0:
                mp=1
            else:
                mp=mp*5
            p[0, tuple([par_loose])] = p[0, tuple([par_loose])] - delta_p
        J_mp= np.vstack((J, mp*np.eye(len(par_loose))))
        r0_mp=np.concatenate((r0_old, np.zeros(len(par_loose))), axis=0)
        delta_p=np.linalg.lstsq(-1*J_mp, r0_mp, rcond=None)[0]
        p[0, tuple([par_loose])] = p[0, tuple([par_loose])] + delta_p
        it+=1
    Curv=J.T @ J
        
    return (p, ssq, C, A, Curv)  
        

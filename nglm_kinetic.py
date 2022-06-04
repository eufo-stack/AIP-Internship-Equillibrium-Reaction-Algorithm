# -*- coding: utf-8 -*-


import numpy as np
from kinfun import kinfun
import time
import math


def nglm_kinetic(t,c,k,n,m,d,x,U,gaus):
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
    par_loose=[0,1] #chooses index of k value to update i.e. if [0] only the first k value would be updated, if [0,1] both would be updated. Current test updates the first value and has the second value already correct and left correct.
    it=0
    J=None
    delta_p=np.zeros((len(par_loose), 1))
    r0_old = np.zeros((190,))
    while it<10:

        k=abs(k)
        #print(k)
        if k[0][1]>100:
          k[0][1]=0.1
          continue
        if k[0][0]>100:
          k[0][0]=0.1
          continue
        s= solve_ivp(kinfun, (1, 20), c,method = 'RK45', t_eval=t,  args=(k, x, U) )

        if J is None:
            J = np.zeros((15390, len(par_loose))) #15390 found through np.matrix.flatten(81,190)
            #print(J)
        C_new=s.y
        #print(C_new)
        #print(C_new.T.shape)
        #print(np.nan_to_num(d.T).shape)
        try:
          g_new=np.linalg.lstsq(np.nan_to_num(d.T),C_new.T)[0] 
          #for i in g_new:
          #  i[2]=0
        except:
          ssq=1000
          Curv=np.array([[1,1],[1,1]])
          #print(Curv.shape)
          print("Dimension error",k)
          break

        D_sim= np.dot(gaus,C_new)
        r0=d-D_sim
    
        ssq = np.linalg.norm(r0 @ r0.T)
      
        conv_crit=(ssq_old-ssq)/(ssq_old)
        diff =np.linalg.norm(d-D_sim)
        #print(diff)

        #if diff<25:
        #     print("break")
        #     break
        if abs(conv_crit)<=mu:
            if mp==0:
                break
            else:
                #print(it)
                mp=0
                r0_old=r0
            
        elif conv_crit>mu:
            mp=mp/3
            ssq_old=ssq
            r0_old=r0
            for i in range(len(par_loose)):
                
                k[0, par_loose[i]]=(1+delta)*k[0, par_loose[i]]
                s2 =solve_ivp(kinfun, (1, 20), c,method = 'RK45', t_eval=t,  args=(k, x, U) )
                C_new2=s2.y
                g_new2=np.linalg.lstsq(np.nan_to_num(d.T),C_new2.T)[0] 
                #for j in g_new2:
                #  j[2]=0
                D_sim2= np.dot(gaus,C_new2)
                r=d-D_sim2
                J[:, i]=np.matrix.flatten((r-r0)/(delta*k[0, par_loose[i]]))
                k[0, par_loose[i]]=k[0, par_loose[i]]/(1+delta) 
        elif conv_crit < -mu:
            if mp==0:
                mp=1
            else:
                mp=mp*5
            k[0, tuple([par_loose])] = k[0, tuple([par_loose])] -delta_p
        J_mp= np.vstack((J, mp*np.eye(len(par_loose),dtype="float64")))
        #print(r0_old.shape)
        r0_mp=np.concatenate((np.matrix.flatten(r0_old), np.zeros(len(par_loose))), axis=0,dtype="float64") #flatten may be the issue?
        #print(J)
        #print(r0_mp.shape)
        
        delta_p=np.linalg.lstsq(-1*J_mp, r0_mp, rcond=None)[0] 
        if delta_p[0]>100:
          break
        if delta_p[1]>100:
          break
        k[0, tuple([par_loose])] = k[0, tuple([par_loose])] +delta_p
        it+=1
        
    Curv=J.T @ J
        
    print(k,U+x,x)
    return s,k,ssq,Curv

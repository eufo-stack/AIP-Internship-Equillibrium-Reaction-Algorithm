# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:51:58 2022


"""

import numpy as np
from kinfun import kinfun
from nglm_kinetic import nglm_kinetic
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from itertools import product


def aikinetic(t,c,k,n,m,d,gaus):
  #colors = ['b','g', 'r','c','m','y','k','w']
  x = product([1, 0], repeat=n*m)
  x = np.reshape(list(x), (-1, n, m))
  #print(x)
  #x=[z for z in x if np.array_equal(z[0],np.array([1,0,0])) and np.array_equal(z[1],np.array([0, 1,0]))] #the correct model only for testing

  Up = product([1, 0], repeat=n*m)
  Up = np.reshape(list(Up), (-1, n, m))
  #Up=[z for z in Up if np.array_equal(z[0],np.array([0, 1,0])) and np.array_equal(z[1],np.array([0, 0,1]))]#^^^^

  cnt = 0
  klim= 3
  clim = 4
  sols=[]

  best_sigk=np.array([1])
  k =np.array([[0.1,0.1]]) #starting out near correct k values for testing
  sig_k=np.array([1])
  for j in range(len(Up)):
    for i in range(len(x)):
      #Up[j][0]=[0, 1,0]
      #x[i][0]=[1,0,0]
      U = Up[j]-x[i]


      s,k_new,ssq,Curv= nglm_kinetic(t,c,k,n,m,d,x[i],U,gaus)
      #k_new=np.round(k_new,2)
      sol= solve_ivp(kinfun, (1, 20), c,method = 'RK45', t_eval=t,  args=(k_new, x[i], U) ) #this isnt using the s used in nglm, recomputes solve_ivp using only the new found k
      C_new=sol.y
      try:
        g_new=np.linalg.lstsq(np.nan_to_num(d.T),C_new.T)[0] 
        #for v in g_new:
        #   v[2]=0
      except:
        print("Dimension error")
        break
      D_sim= np.dot(gaus,C_new)

      diff=np.linalg.norm(D_real-D_sim)

      sig_r = np.sqrt(ssq/(np.prod(np.shape(np.matrix.flatten(D_real))) - len(k)));
      #print("sig_r",sig_r)
      print("Diff",diff)
      #if diff<100:
      #  break
      try:
          sig_k = sig_r * np.sqrt(np.diag(np.linalg.inv(Curv)));
          #print("Curv",Curv.shape)	
      except:
          print('Error getting sig_k: ', k)

    
      if np.linalg.norm(sig_k, 2)<np.linalg.norm(best_sigk, 2):
        if k_new[0][0]<10 and k_new[0][1] <10:
          print("New K:",k_new)
          best_sigk=sig_k
          k=k_new

      sols.append((sol.y,sol.t,Up[j],x[i],D_sim,k_new))
      #print(k_new)


      cnt+=1
  c=np.append(c, [0], axis=0)        
  return sols
                

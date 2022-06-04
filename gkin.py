# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:26:02 2022

@author: sheld
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import RK45
from AiKinetic import aikinetic
from kinfun import kinfun
import matplotlib.pyplot as plt

t = np.arange(1, 20, 0.1)
#c = np.array([1, 0, 0])
c=np.array([1,0,0])
k =np.array([[0.4,0.2]])
#Xr=np.zeros((c.size,k.size))
Xr=np.array([[1,0,0],[0, 1,0]], dtype=np.float64)
#Xp=np.zeros((c.size,k.size))
Xp=np.array([[0, 1,0],[0, 0,1]], dtype=np.float64)
#Xr=np.array([[1, 0, 0],[0, 1, 0]], dtype=np.float64)
#Xp=np.array([[0, 1, 0],[0, 0, 1]], dtype=np.float64)
X=Xp-Xr


sol=  solve_ivp(kinfun, (1, 20), c,method = 'RK45', t_eval=t,  args=(k, Xr, X) )



x=np.arange(300,701,5)
x=x.T
t_mean=[450, 480, 495] 
t_std=[30, 28, 45] 
height =[1, 0.8, 1.5]
C=sol.y
gaus=np.zeros((81,3))
for i in np.arange(2):
   for j in range(len(x)):
      A=height[i];
      Xbar=t_mean[i];
      gaus[j,i]=(A*(np.exp(-(np.power((x[j]-Xbar),2))/(2*(t_std[i])**2))));

D=np.dot(gaus, C) # should have 81xtime dimension 

D_real = np.nan_to_num(D+ np.random.randn(81, len(sol.t))* 0.00000000000000000000001); #very small number, deleting left of + didnt seem to matter



n=2#c.size
m=3#k.size

#replace g with D_real, generate new C, obtain g again using least square with input D_real and generated C -> multiply again=dot(gaus,C) -> calculate diff 

sols_test=aikinetic(t,c,k,n,m,D_real,gaus)


diffs=[]
for i in range(len(sols_test)):
  if len(sols_test[i][0])==len(c):
    d_diffs=np.linalg.norm(D_real-sols_test[i][4])
    diffs.append((d_diffs,sols_test[i][0],sols_test[i][1],sols_test[i][2],sols_test[i][3],sols_test[i][5]))
  
diffs.sort(key = lambda x: x[0])
top5=[(x[0],x[3],x[4],x[5]) for x in diffs][:5]






print("Top 5: ",top5)

  

print("Top: "," Diff:",top5[0][0]," XP:",top5[0][1]," XR:",top5[0][2]," K:",top5[0][3])


#print(diffs)
#plt.plot
#plt.plot(diffs[0][2], diffs[0][1][0], 'r')
#plt.plot(diffs[0][2], diffs[0][1][1], 'b')
#plt.plot(diffs[0][2], diffs[0][1][2], 'g')
#plt.plot(sol.t, sol.y[0], 'r')
#plt.plot(sol.t, sol.y[1], 'b')
#plt.plot(sol.t, sol.y[2], 'g')
# print(sol.t)
# print(sol.y)
# print(sol.y.shape)
# print(sol.y)
# print(sol.y.shape)

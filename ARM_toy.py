from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
import sys
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
import os

#%%

def fun(b,t):
    return np.square(b-t)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


t = np.array([0.49,0.499,0.501,0.51])

#Monte carlo samples for REINFORCE/ARM grad
K1=1
K2=1
J=len(t);

#record ARM
phi_arm = np.zeros([J])
grad_record_a = []
prob_record_a = []

#record REINFORCE
phi_r = np.zeros([J])
grad_record_r = []
prob_record_r = []

#record True gradient
phi_true = np.zeros([J])
grad_record_true = []
prob_record_true = []

IterMax=3000;
stepsize = 1;

for iter in range(IterMax):
   ####################################
   
   #ARM
   u = np.random.uniform(size=[K1,J])
   E1 = (u>sigmoid(-phi_arm)).astype('float32')
   E2 = (u<sigmoid(phi_arm)).astype('float32')
   F1 = fun(E1,t)
   F2 = fun(E2,t)
   grad_arm = np.mean((F1-F2)*(u-1/2),axis=0)
   phi_arm = phi_arm + stepsize * grad_arm
   prob_record_a.append(sigmoid(phi_arm))
   grad_record_a.append(grad_arm)
   p_arm = sigmoid(phi_arm)
   
   
   ####################################
   
   #REINFORCE
   p_r = sigmoid(phi_r)
   b_r = np.random.binomial(1,p_r,size=[K2,J])
   
   grad_r = np.mean(fun(b_r,t)*(b_r*(1-p_r)-p_r*(1-b_r)),axis=0)
   phi_r = phi_r + stepsize * grad_r
   prob_record_r.append(p_r)
   grad_record_r.append(grad_r)
   
   ####################################
   
   #True Gradient
   p_true = sigmoid(phi_true)
   grad_true = (p_true*(1-p_true))*(fun(1,t)-fun(0,t))
   phi_true = phi_true + stepsize * grad_true
   prob_record_true.append(p_true)
   grad_record_true.append(grad_true)



all_ = [grad_record_a,grad_record_r,grad_record_true,prob_record_a,prob_record_r,prob_record_true]
[grad_record_a,grad_record_r,grad_record_true,prob_record_a,prob_record_r,prob_record_true] = \
    [np.array(_) for _ in all_]    


#plot every 10 points   
idx = np.arange(0, grad_record_a.shape[0],10)
grad_record_a = grad_record_a[idx,:]
grad_record_r = grad_record_r[idx,:]
grad_record_true = grad_record_true[idx,:]    

   
f, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3,sharex=True,figsize=(15,15))  

ax1.plot(idx,grad_record_true)
ax1.set_ylabel('Gradient',fontsize='x-large') 
ax1.set_title('True_grad',fontsize='x-large')

ax2.plot(idx,grad_record_r)
ax2.set_title('REINFORCE_grad',fontsize='x-large')

ax3.plot(idx,grad_record_a)
box = ax3.get_position()
ax3.set_position([box.x0, box.y0, box.width, box.height])
ax3.legend([str(_) for _ in t],loc='center left', bbox_to_anchor=(1, 0.5))
ax3.set_title('ARM_grad',fontsize='x-large')

ax4.plot(prob_record_true)
ax4.set_ylabel('Probability',fontsize='x-large') 
ax4.set_xlabel('Iteration',fontsize='x-large')
ax4.set_title('True_prob',fontsize='x-large')


ax5.plot(prob_record_r)
ax5.set_xlabel('Iteration',fontsize='x-large')
ax5.set_title('REINFORCE_prob',fontsize='x-large')


ax6.plot(prob_record_a)
box = ax6.get_position()
ax6.set_position([box.x0, box.y0, box.width, box.height])
ax6.legend([str(_) for _ in t],loc='center left', bbox_to_anchor=(1, 0.5))
ax6.set_xlabel('Iteration',fontsize='x-large')
ax6.set_xlabel('Iteration',fontsize='x-large')
ax6.set_title('ARM_prob',fontsize='x-large')


plt.tight_layout()
plt.show()
 
















   

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 01:37:29 2022

@author: lixuy
"""

import numpy as np
import torch
from PINN import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

#plt.style.reload_library()
#plt.style.use('science')
#plt.style.use(['science','no-latex'])

u_true = np.loadtxt("y.txt", dtype='float32', delimiter='\t') * 1e2

u_true_1d = u_true.T.flatten()[:, None]

Nx = 16
Nt = 16*10
L = 0.4
tmax = 0.045

x = np.linspace(0, L, Nx+2)[1:-1].reshape(-1, 1) # take only the middle 16 elements
t = np.linspace(0, tmax, Nt).reshape(-1, 1)
X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
X_T = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data


x_lb = np.array([0]).reshape(-1, 1)
x_ub = np.array([L]).reshape(-1, 1)
t_bc = np.linspace(0, tmax, Nt).reshape(-1, 1)

X_bc_lb, T_bc_lb = np.meshgrid(x_lb, t_bc)
X_bc_ub, T_bc_ub = np.meshgrid(x_ub, t_bc)
X_T_bc_lb = np.hstack((X_bc_lb.flatten()[:, None], T_bc_lb.flatten()[:, None]))
X_T_bc_ub = np.hstack((X_bc_ub.flatten()[:, None], T_bc_lb.flatten()[:, None]))

def set_seed(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(10)
length = X_T.shape[0]
ratio = 0.2
idx = np.random.choice(length, int(length*ratio), replace=False)
X_T_train = X_T[idx,:]
u_true_1d_train = u_true_1d[idx,:]


L_f = 1e-13 # 0
#7e-9 #8e-09 #2e-8 #1e-08
L_bc = 4e-14 # 5e-14
#30e-5 #1.6e4 #4e-5 #4e-05

layers = np.array([2, 50, 50, 50, 50, 1])
activation = 'tanh'
optimizer_name = 'LBFGS'
model = PhysicsInformedNN_pbc(X_T_train, X_T_bc_lb, X_T_bc_ub, u_true_1d_train, optimizer_name, layers, L_f = L_f, L_bc = L_bc)

model.train()


#---------data extrapolate
x = np.linspace(0, L, Nx+2)[1:-1].reshape(-1, 1)  # take only the middle 16 elements
t2 = np.linspace(0, 2*tmax, 2*Nt).reshape(-1, 1)
X2, T2 = np.meshgrid(x, t2) # all the X grid points T times, all the T grid points X times
X_T2 = np.hstack((X2.flatten()[:, None], T2.flatten()[:, None])) # all the x,t "test" data

u_pred_1d = model.predict(X_T)
u_pred = u_pred_1d.reshape(160,16).T*1e1

u_pred_1d_extrapolate = model.predict(X_T2)
u_pred2 = u_pred_1d_extrapolate.reshape(320,16).T*1e1


#-----------heatmap
import matplotlib.ticker as ticker

y = np.loadtxt("y.txt", dtype='float32', delimiter='\t') * 1e3
y2 = np.loadtxt("y-extrapolate.txt", dtype='float32', delimiter='\t') * 1e3


#true
fig, ax = plt.subplots(figsize = (6,4))
im = plt.imshow(y.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("True", fontsize=20)
plt.xlabel('x', fontsize=20)
plt.ylabel('t', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=16)


#pred
fig, ax = plt.subplots(figsize = (6,4))
im = plt.imshow(u_pred.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("Prediction", fontsize=18)
plt.xlabel('x', fontsize=16)
plt.ylabel('t', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=14)
plt.show()

#error
fig, ax = plt.subplots(figsize = (6,4))
im = plt.imshow(abs(u_pred-y).T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("Error", fontsize=18)
#plt.xlabel('Space/(m)', fontsize=16)
#plt.ylabel('Time/(s)', fontsize=18)
plt.xlabel('x', fontsize=16)
plt.ylabel('t', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=14)
plt.show()


#---extrapolate heatmap
#pred
fig, ax = plt.subplots(figsize = (6,7))
im = plt.imshow(u_pred2.T, cmap='jet', aspect='auto',extent=[0,L,2*tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("Prediction Extrapolate", fontsize=18)
plt.xlabel('x', fontsize=16)
plt.ylabel('t', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=14)
plt.show()


#error
fig, ax = plt.subplots(figsize = (6,7))
im = plt.imshow(abs(u_pred2-y2).T, cmap='jet', aspect='auto',extent=[0,L,2*tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("Error Extrapolate", fontsize=18)
#plt.xlabel('Space/(m)', fontsize=16)
#plt.ylabel('Time/(s)', fontsize=18)
plt.xlabel('x', fontsize=16)
plt.ylabel('t', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=14)
plt.show()




error1 = np.mean(abs(u_pred-y))
error2 = np.mean(abs(u_pred2-y2))

print("error1: ",error1)
print("error2: ",error2)
print('Lf: ',L_f)
print('Lbc: ',L_bc)


#np.savetxt('PINN-pred.txt',u_pred, delimiter='\t')
#np.savetxt('PINN-pred2.txt',u_pred2, delimiter='\t')

np.save('PINN-pred',u_pred)
np.save("PINN-pred2",u_pred2)


# Execution time
t12 = np.linspace(tmax, 2*tmax, Nt).reshape(-1, 1)
X12, T12 = np.meshgrid(x, t12)
X_T12 = np.hstack((X12.flatten()[:, None], T12.flatten()[:, None]))

import time
st = time.time()
N=500
for i in range(N):
    u_pred_1d_extrapolate = model.predict(X_T)
    u_pred1 = u_pred_1d_extrapolate.reshape(-1,16).T*1e1
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time/N, 'seconds')


st = time.time()
N=500
for i in range(N):
    u_pred_1d_extrapolate = model.predict(X_T12)
    u_pred12 = u_pred_1d_extrapolate.reshape(-1,16).T*1e1
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time/N, 'seconds')




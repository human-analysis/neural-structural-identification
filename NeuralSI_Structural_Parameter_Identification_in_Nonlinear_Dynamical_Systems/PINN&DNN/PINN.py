# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 00:18:33 2022

@author: lixuy
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from choose_optimizer import *

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DNN(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = Sine
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    


class PhysicsInformedNN_pbc():
    def __init__(self, X_T, X_T_bc_lb, X_T_bc_ub, u_true_1d, optimizer_name, layers, L_f=1, lr=1, L_bc=1, activation='tanh',):
        self.x_u = torch.tensor(X_T[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_T[:, 1:2], requires_grad=True).float().to(device)
        
#        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
#        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(X_T_bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(X_T_bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(X_T_bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(X_T_bc_ub[:, 1:2], requires_grad=True).float().to(device)
        
        self.dnn = DNN(layers, activation).to(device)
        self.u = torch.tensor(u_true_1d, requires_grad=True).float().to(device)

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)
        self.iter = 0
        self.force = 1#000
        self.EI = 36.458332
        self.damp = 25
        self.rhoA = 0.675
        self.L_f = L_f
        self.L_bc = L_bc
        self.mae = nn.L1Loss()

    def net_u(self, x, t):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]
        u_xxx = torch.autograd.grad(
            u_xx, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]
        u_xxxx = torch.autograd.grad(
            u_xxx, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f = self.rhoA*u_tt + self.EI*u_xxxx + self.damp*u_t - self.force
        return f

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]
        
        u_lb_xx = torch.autograd.grad(
            u_lb_x, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]
        

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_xx = torch.autograd.grad(
            u_ub_x, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_xx, u_ub_xx

    def loss_pinn(self, verbose=True):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()

        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub)
        u_pred_lb_xx, u_pred_ub_xx = self.net_b_derivatives(u_pred_lb, u_pred_ub, 
                                                            self.x_bc_lb, self.x_bc_ub)
        u_pred = self.net_u(self.x_u, self.t_u)
        
        
        loss_u = torch.mean(torch.abs(self.u - u_pred)  )
        loss_bc = torch.mean(torch.abs(u_pred_lb)  ) + torch.mean(torch.abs(u_pred_ub)  ) + \
                    torch.mean(torch.abs(u_pred_lb_xx)  ) + torch.mean(torch.abs(u_pred_ub_xx)  )
        loss_pde = torch.mean(torch.abs(self.net_f(self.x_u, self.t_u))  )
        loss = loss_u + self.L_f * loss_pde + self.L_bc * loss_bc
        
        if loss.requires_grad:
            loss.backward()

        grad_norm = 0
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 50 == 0:
                print(
                    'epoch %d, gradient: %.3e, loss: %.3e, loss_u: %.5e, loss_bc: %.3e, loss_pde: %.3e' \
                    % (self.iter, grad_norm, loss.item(), loss_u.item(), loss_bc.item(), loss_pde.item())
                )
            self.iter += 1
#        print(loss)
        return loss

    def train(self):
        self.dnn.train()
        self.optimizer.step(self.loss_pinn)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        
        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()
        return u






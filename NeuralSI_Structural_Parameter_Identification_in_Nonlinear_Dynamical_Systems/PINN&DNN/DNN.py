# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:45:08 2022

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
    


class NN_pbc():
    def __init__(self, X_T, u_true_1d, optimizer_name, layers, lr=1, activation='tanh',):
        self.x_u = torch.tensor(X_T[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_T[:, 1:2], requires_grad=True).float().to(device)
        
        
        self.dnn = DNN(layers, activation).to(device)
        self.u = torch.tensor(u_true_1d, requires_grad=True).float().to(device)

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)
        self.iter = 0
        self.mae = nn.L1Loss()

    def net_u(self, x, t):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def lossNN(self, verbose=True):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        
        loss_u = torch.mean(torch.abs(self.u - u_pred)  )
        loss = loss_u 
        
        if loss.requires_grad:
            loss.backward()

        grad_norm = 0
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 20 == 0:
                print(
                    'epoch %d, gradient: %.3e, loss: %.3e' \
                    % (self.iter, grad_norm, loss.item())
                )
            self.iter += 1
#        print(loss)
        return loss

    def train(self):
        self.dnn.train()
        self.optimizer.step(self.lossNN)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        
        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()
        return u


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:02:40 2023

@author: dr
"""

from .SmoothnessIndex import Kalhor_SmoothnessIndex



# ===========import torch
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_random_data(n_data, dim_input, dim_target, n_data_test):
    inp = torch.randn(n_data, dim_input, device=(device))
    target = torch.randn(n_data, dim_target, device=(device))

    inp_test = torch.randn(n_data_test, dim_input, device=(device))
    target_test = torch.randn(n_data_test, dim_target, device=(device))

    return inp, target, inp_test, target_test


def toy_example(n_data, dim_input, dim_target, n_data_test):
    # sinc function
    inp = torch.randn(n_data, dim_input, device=(device))
    target = torch.sin(inp[:, 0:1] * inp[:, 1:]) / (1e-5 + inp[:, 0:1] * inp[:, 1:]) #* (
                # 1 + 0.1 * torch.randn(n_data, 1, device=(device)))

    inp_test = torch.randn(n_data_test, dim_input, device=(device))
    target_test = torch.sin(inp_test[:, 0:1] * inp_test[:, 1:]) / (1e-5 + inp_test[:, 0:1] * inp_test[:, 0:1])

    return inp, target, inp_test, target_test


# ==========================================  SmI
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
example = 2
# Example1====  generate_random_data
if example == 1:
    n_data = 1000
    dim_input = 10
    dim_target = 2
    n_data_test = 100
    inp, target, inp_test, target_test = generate_random_data(n_data, dim_input, dim_target, n_data_test)
# Example===== two dimensional sinc function
if example == 2:
    n_data = 1000
    dim_input = 2
    dim_target = 1
    n_data_test = 50
    inp, target, inp_test, target_test = toy_example(n_data, dim_input, dim_target, n_data_test)

instant = Kalhor_SmoothnessIndex(inp, target)
print('===== SmI========')

smi = instant.smi_linear()
print('Linear Smoothness Index is: ', smi.detach().cpu().numpy())
gama = 1
smi = instant.smi_exp(gama)
print('Exponential Smoothness Index is: ', smi.detach().cpu().numpy())

# ================== high order SmI
order = 2
print('=====high order SmI========', 'order=', order)
smi = instant.high_order_smi_linear(order)
print('High Smoothness Index with order', order, ' is:', smi.detach().cpu().numpy())
gama = 1
smi = instant.high_order_smi_exp(order, gama)
print('High Smoothness exp Index with order', order, ' is:', smi.detach().cpu().numpy())

# ================== Anti SmI
order = 1
print('=====Anti SmI========', 'order=', order)
smi = instant.anti_smi_linear(order)
print('Anti Smoothness Index with order', order, ' is:', smi.detach().cpu().numpy())
gama = 1
smi = instant.anti_smi_exp(order, gama)
print('Anti Smoothness exp Index with order', order, ' is:', smi.detach().cpu().numpy())

# ================== soft SmI

order = 2
print('=====soft SmI========', 'order=', order)
smi = instant.soft_order_smi_linear(order)
print('Soft linear Smoothness Index with order', order, ' is:', smi.detach().cpu().numpy())
smi = instant.soft_order_smi_exp(order, gama)
print('Soft Smoothness exp Index with order', order, ' is:', smi.detach().cpu().numpy())
# ================== cross SmI
print('=====cross SmI========')
cr_smi = instant.cross_smi_linear(inp_test, target_test)
print('Cross Smootheness Index is: ', cr_smi.detach().cpu().numpy())

# ================== local SmI
print('=====Local SmI========')
n_neighb=25
l_smi, ancher, postitive, negative = instant.triplet_local_smi(n_neighb)
print('Local Smootheness Index is: ', l_smi.detach().cpu().numpy())

# ================== global SmI
print('=====global SmI========')
g_smi, ancher, postitive, negative = instant.triplet_global_smi()
print('Global Smootheness Index is: ', g_smi.detach().cpu().numpy())

# ================== signle_input_smi_linear
print('=====signle_input_smi_linear========')
signle_input_smi_linear = instant.signle_input_smi_linear()
print('the smi score of each input is: ', signle_input_smi_linear.detach().cpu().numpy())

# ================== forward_ranking_smi
print('=====forward_ranking_smi========')

smi_ranked_inputs, ranked_inputs = instant.forward_input_ranking_smi_linear()
print('Ranked  inputs form most important to least important  are: ', ranked_inputs)
print('smi for involving Inputs are: ', smi_ranked_inputs.detach().cpu().numpy())

# ================== forward_selection_smi

print('=====forward_input_ranking_by_smi========')

disturbance = torch.randn(n_data, 3, device=(device)) * 10
inp_disturbance = torch.concat((inp, disturbance), 1)

instance_disturbance = Kalhor_SmoothnessIndex(inp_disturbance, target)

smi_ranked_inputs, ranked_inputs = instance_disturbance.forward_input_ranking_smi_linear()

plt.plot(smi_ranked_inputs.cpu().detach().numpy(), 'b')
plt.xlabel("Number of selected inputs")
plt.ylabel("SmI")
plt.title("Forward input ranking by SmI")
smi_ranked_inputs = torch.transpose(smi_ranked_inputs, 0, 1)

print('Ranked features are: ', ranked_inputs)
print('smi for the best chosen Inputs are: ', smi_ranked_inputs.detach().cpu().numpy())
# (9)================== ranked_features_best

print('=====ranked_features_best========')
input_best_smi, ranked_inputs_best = instance_disturbance.get_best_inputs_forward_smi_linear()

instance_best = Kalhor_SmoothnessIndex(input_best_smi, target)

print('Sepration Index for input_best_smi is: ', instance_best.smi_linear().detach().cpu().numpy())
print('the best feachers are: ', ranked_inputs_best.t())

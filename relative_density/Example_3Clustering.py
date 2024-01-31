#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:02:40 2023

@author: dr
"""
# [0] Import libraries and modules===============================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from mmt_3clustering import Kalhor_Clustering
from mmt_3clustering import module_clustering

def generate_data(n_data_cluster, dim_data, n_clusters, dim_box):
    data_3d=torch.randn(n_data_cluster,dim_data, n_clusters, device=device)
    center_3d=torch.randn(1,dim_data,n_clusters, device=device).repeat([n_data_cluster,1,1])*dim_box
    data=(data_3d+center_3d).reshape([n_data_cluster*n_clusters ,dim_data])
    return data

# [1] Determine the device=======================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# [2] load dataset and choose options  ========================================================

case = 1
# Example1====  generate_random_data
if case == 1:
    n_clusters =20+torch.randperm(100)[0]+1
    print('real number of clusters is:', n_clusters.detach().cpu().numpy())
    dim_data = 2
    n_data_cluster = 100
    dim_box=40
    data= generate_data(n_data_cluster, dim_data, n_clusters,dim_box)
#=============================================================================================
# [3]  choose options and call module_clustering==============================================
black_hole=False
normalize_do=False
n_cluster_h, label_data, center_clusters =module_clustering(data,normalize_do=normalize_do)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:02:40 2023

@author: dr
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from mmt_4feature_selection_roudbari import Feature_Selection_Unsueprvised
from mmt_4feature_selection_roudbari import module_feature_selection_unsupervised





# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#===================== choose cases and feature_selection_type
case = 3
# [2] classification_with_labels
#[3] unsupervised
# First by forward selection choose inputs which has lowest smi with each other( to remove those which have high correlations with eachother)
# Then by a forward selection select some of them which makes maximum number of clusters
if case == 3:
    feature_selection_type='unsupervised'
    #========================================================================================
    def generate_data(n_data_cluster, dim_data, n_clusters, dim_box):
        data_3d=torch.randn(n_data_cluster,dim_data, n_clusters, device=device)
        center_3d=torch.randn(1,dim_data,n_clusters, device=device).repeat([n_data_cluster,1,1])*dim_box
        data=(data_3d+center_3d).reshape([n_data_cluster*n_clusters ,dim_data])
        n_data_total=n_data_cluster*n_clusters
        dim_noise=5
        noise=torch.randn(n_data_total,dim_noise, device=device)
        data=torch.cat((data,data,-0.5*data[:,0:1],-data[:,1:].sin(), noise),1)
        return data
    #========================================================================================
    n_clusters =30+torch.randperm(30)[0]+1
    print('real number of clusters is:', n_clusters.detach().cpu().numpy())
    dim_data = 3
    n_data_cluster = 100
    dim_box=5
    features= generate_data(n_data_cluster, dim_data, n_clusters,dim_box)


#===============================================================
normalize_do=True
if feature_selection_type=='unsupervised':
     best_features_set, data_best =module_feature_selection_unsupervised(features,normalize_do=normalize_do)

print('best_features_set are:', best_features_set.detach())

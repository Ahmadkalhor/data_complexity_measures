#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:02:40 2023

@author: dr
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
#
from ldi_data_scoring import Data_Scoring_Unsupervised
from ldi_data_scoring import module_data_scoring





# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#===================== choose cases and feature_selection_type

scoring_type='unsupervised'
#========================================================================================
def generate_data(n_data_cluster, dim_data, n_clusters, dim_box):
    data_3d=torch.randn(n_data_cluster,dim_data, n_clusters, device=device)
    center_3d=torch.randn(1,dim_data,n_clusters, device=device).repeat([n_data_cluster,1,1])*dim_box
    data=(data_3d+center_3d).reshape([n_data_cluster*n_clusters ,dim_data])
    return data
#========================================================================================
n_clusters =30+torch.randperm(30)[0]+1
print('real number of clusters is:', n_clusters.detach().cpu().numpy())
dim_data = 2
n_data_cluster = 100
dim_box=20
features= generate_data(n_data_cluster, dim_data, n_clusters,dim_box)
n_max_clusters=2*n_clusters

#===============================================================
normalize_do=True
print('scoring_type: ', scoring_type)
data_scores =module_data_scoring(features, n_max_clusters,normalize_do=normalize_do)
    #========================================================================================

print('data_scores',data_scores)


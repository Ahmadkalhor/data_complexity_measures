#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:02:40 2023

@author: dr
"""

from LinearDensityIndex import Kalhor_LinearDensityIndex

import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_data_cluster, dim_data, n_clusters):
    data_3d=torch.randn(n_data_cluster,dim_data, n_clusters, device=device)*0.3
    # center_3d=torch.zeros(n_data_cluster,dim_data, n_clusters, device=device)
    # for k in range(n_clusters):
    #     center_3d[:,:,k]=center_3d[:,:,k]+k*10
    center_3d=torch.randn(1,dim_data,n_clusters, device=device).repeat([n_data_cluster,1,1])*5
    data=(data_3d+center_3d).reshape([n_data_cluster*n_clusters ,dim_data])
    return data
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
example = 1
# Example1====  generate_random_data
if example == 1:
    n_clusters =torch.randperm(30)[0]+1
    print('real number of clusters is:', n_clusters.detach().cpu().numpy())
    dim_data = 2
    n_data_cluster = 500
    data= generate_data(n_data_cluster, dim_data, n_clusters)
# =================================print('===== linear_density_clustering========')
instant = Kalhor_LinearDensityIndex(data)

n_max_clusters=2*n_clusters
print('n_max_clusters', n_max_clusters)
kmeans_repeat=20
n_cluster_h, label_data, av_lin_den,sum_lin_den, center_clusters = instant.ldi_clustering(n_max_clusters, kmeans_repeat)
print('Number of clusters is: ', n_cluster_h)
print('sum of linear density is: ', sum_lin_den.detach().cpu().numpy())
av_lin_den=av_lin_den.cpu()
# =================================print('===== linear_density_clustering========')
print('label_data',label_data.min(), label_data.max())
data_score=instant.data_score_unsupervised(label_data)
print('data_score',data_score.min())


if dim_data==2:
    data=data.cpu()
    label_data=label_data.cpu()
    center_clusters=center_clusters.cpu()
    for k_cluster in range(n_cluster_h):
        value, indices=((label_data==k_cluster)*1).sort()
        data_cluster=data[indices[value.argmax(0):,0],:]
        plt.subplot(2,1,1)
        plt.plot(data_cluster[:,0], data_cluster[:,1], '.b')
        plt.plot(center_clusters[k_cluster,0], center_clusters[k_cluster,1], '*r')
    plt.subplot(2,1,2)
    plt.plot(av_lin_den, '*r')
    plt.show()
else:
    if dim_data==1:
        data=data.cpu()
        label_data=label_data.cpu()
        center_clusters=center_clusters.cpu()
        for k_cluster in range(n_cluster_h):
            value, indices=((label_data==k_cluster)*1).sort()
            data_cluster=data[indices[value.argmax(0):],:]
            y_axis=torch.ones(len(data_cluster),1, device='cpu')
            plt.subplot(2,1,1)
            plt.plot(data_cluster,y_axis, '.b')
            plt.plot(center_clusters[k_cluster,0],1, '*r')
        plt.subplot(2,1,2)
        plt.plot(av_lin_den, '*r')
        plt.show()
    else:
        plt.plot(av_lin_den, '*r')
        plt.show()

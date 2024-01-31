
"""
Created on Mon Jun 26 17:11:36 2023
This code was written by Dr. Ahmad Kalhor (ad.kalhor@gmail.com)in pytorch framework
It includes "Kalhor_LinearDensityIndex" class with 4 methods.
"""
# ===========Some notes about the class
# All methods of the class are developed from the concept of Linear Density Index(LDI).
# LDI denotes as an unsupervised data complexity measure which is utilised to analyze and design AI-models in feature representation problems.
# To illustrate  how one can use this class, four illustrative examples are provided in four executable files.

# ==========Some applications of using "Kalhor_LinearDensityIndex" class:
# 1-To detect clusters in a distribution and get its sum of linear density
# 2-To eliminate features which do not change the number of clusters when all features exist
# 3- Using a few shot labeled data, To label unsupervised data

import torch
import numpy as np
import matplotlib.pyplot as plt
# ========================start of class=======================================
class Kalhor_Clustering:
    def __init__(self, data):
        self.device1 = data.device
        # --------- target normalization
        self.big_number = 1e10
        self.n_data = data.shape[0]
        self.dim_data = data.shape[1]
        self.dis_matrix_input = torch.cdist(data, data, p=2).fill_diagonal_(self.big_number)
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.data=data

    # (1. nomalization method)==========================================
    def get_normalization_matrix(self):
        self.small_number = 1e-10
        mean_data = self.data.mean(0)
        std_data = (self.data.std(0) + self.small_number).reshape(self.dim_data)
        scale_matrix=torch.diag(1/std_data)
        return mean_data, scale_matrix
    # (2. ldi_clustering method)==========================================
    def ldi_clustering(self, n_cluster_max,kmeans_repeat):

        def get_param( label_data, k_cluster):
            value, indices=((label_data==k_cluster)*1).sort()
            inp=self.data[indices[value.argmax(0):],:]
            center=inp.mean(0)
            cov=torch.cov(inp.T)
            num_cluster=inp.shape[0]
            if self.dim_data>1:
                Eig_vec, Eig_val, V=cov.svd()
                lin_den=num_cluster/Eig_val[0]
                div_score=num_cluster*Eig_val[0]
                div_lin=Eig_vec[:,0]
                bias_lin=-center@div_lin
            else:
                lin_den=num_cluster/cov
                div_score=num_cluster*cov
                div_lin=1
                bias_lin=-center

            return div_lin, bias_lin, div_score, lin_den, center, num_cluster
        #===================================================================
        lin_den=torch.zeros(n_cluster_max, device=self.device1)
        div_score=torch.zeros(n_cluster_max, device=self.device1)
        av_lin_den=torch.zeros(n_cluster_max, device=self.device1)
        num_clusters=torch.zeros(n_cluster_max, device=self.device1)
        center_clusters=torch.zeros(n_cluster_max,self.dim_data, device=self.device1)
        div_lin=torch.zeros(self.dim_data,n_cluster_max, device=self.device1)
        bias_lin=torch.zeros(n_cluster_max,1, device=self.device1)
        mean_2=torch.zeros(2,self.dim_data, device=self.device1)
        label_data=torch.zeros(self.n_data, device=self.device1 )
        # First Cluster
        n_cluster=1
        num_clusters[0]=self.n_data
        div_lin[:,0], bias_lin[0], div_score[0], lin_den[0], center_clusters[0,:], num_clusters[0]=get_param(label_data, 0)
        av_lin_den[0]=lin_den[0:n_cluster].mean()
        av_lin_den_max=av_lin_den[0]
        sum_lin_den=av_lin_den[0]
        label_data_best=label_data
        center_clusters_best=center_clusters[0:n_cluster,:]
        n_cluster_best=n_cluster
        k_cluster=0
        stp=0
        while stp==0:

            #Find the worst cluster
            vmax, k_worst=div_score.max(0)
            #-----------divide the worst into two clusters
            value, indices=((label_data==k_worst)*1).sort()
            arg_worst=indices[value.argmax(0):]
            n_cluster=k_cluster+1
            worst_data=self.data[arg_worst,:]
            adr=((worst_data@div_lin[:,k_worst]+bias_lin[k_worst])>0)*1
            value, arg_sort=adr.sort()
            aworst=value.argmax(0)
            arg_worst_up=arg_sort[0:aworst]
            arg_worst_down=arg_sort[aworst:]
            mean_2[0,:]=worst_data[arg_worst_up,:].mean(0)
            mean_2[1,:]=worst_data[arg_worst_down,:].mean(0)
            #================Kmeans clustering
            for km in range(kmeans_repeat):
                dis_2=torch.cdist( mean_2, worst_data , p=2)
                val, arg_sort=((dis_2[0,:]<dis_2[1,:])*1).sort()
                aworst=val.argmax(0)
                arg_worst_up = arg_sort[0:aworst]
                arg_worst_down = arg_sort[aworst:]
                mean_2[0,:] = worst_data[arg_worst_up,:].mean(0)
                mean_2[1,:] = worst_data[arg_worst_down,:].mean(0)
            #==============================================
            if (len(arg_worst_down)>1)&(len(arg_worst_up)>1):
                label_data[arg_worst[arg_worst_down]]=k_cluster
                label_data[arg_worst[arg_worst_up]]=k_worst.int().item()
                center_clusters[k_worst,:]=mean_2[0,:]
                center_clusters[k_cluster,:]=mean_2[1,:]
                div_lin[:,k_worst], bias_lin[k_worst], div_score[k_worst], lin_den[k_worst], center_clusters[k_worst,:], num_clusters[k_worst]=get_param(label_data, k_worst)
                div_lin[:,k_cluster], bias_lin[k_cluster], div_score[k_cluster], lin_den[k_cluster], center_clusters[k_cluster,:], num_clusters[k_cluster]=get_param(label_data, k_cluster)
                av_lin_den[k_cluster]=lin_den[0:n_cluster].mean()
                if self.dim_data==1:
                    av_lin_den[k_cluster]=lin_den[0:n_cluster].mean()/n_cluster
                if (av_lin_den[k_cluster] > av_lin_den_max):
                    av_lin_den_max=av_lin_den[k_cluster]
                    label_data_best=label_data.clone()
                    center_clusters_best=center_clusters[0:n_cluster,:]
                    sum_lin_den=av_lin_den[k_cluster]*n_cluster
                    n_cluster_best=n_cluster
            else:
                stp=1
            if stp==0:
                k_cluster+=1
            if (k_cluster>=n_cluster_max):
                stp=1
        return n_cluster_best, label_data_best, av_lin_den[0:k_cluster],sum_lin_den, center_clusters_best
    #==========
    def kmeans_clustering(self,kcluster,kmeans_repeat):
        rand1=torch.randperm(self.n_data, device=self.device1)
        center_clusters_best=self.data[rand1[0:kcluster],:]
        for km in range(kmeans_repeat):
            dis_2=torch.cdist( center_clusters_best, self.data , p=2)
            values_min, indices_min=dis_2.min(0)
            for j in range(center_clusters_best.shape[0]):
                values_sort, indices_sort=((indices_min==j)*1.).sort()
                argmax=values_sort.argmax()
                center_clusters_best[j:j+1,:]=self.data[indices_min[indices_sort[argmax:]],:].mean(0)
        return center_clusters_best

# =======================End of Class=================================
def module_clustering(data, normalize_do=False):

    device = data.device
    print('data', data.shape)
    kmeans_repeat=100
    dim_data=data.shape[1]
    n_data=data.shape[0]
    if normalize_do:
        instant_initial = Kalhor_Clustering(data)
        mean_data, scale_matrix=instant_initial.get_normalization_matrix()
        shift_matrix=mean_data.repeat([n_data,1])
        data_norm=(data-shift_matrix)@scale_matrix
    else:
        mean_data=torch.zeros(dim_data, device=device)
        scale_matrix=torch.eye(dim_data, device=device)
        shift_matrix=mean_data.repeat([n_data,1])
        data_norm=(data-shift_matrix)@scale_matrix
    n_max_clusters=round(n_data/2)
    #=============================================
    instant = Kalhor_Clustering(data_norm)
    n_cluster_h, label_data, av_lin_den,sum_lin_den, center_clusters = instant.ldi_clustering(n_max_clusters, kmeans_repeat)
    print('the predicted number of clusters is:',n_cluster_h )
    shift_matrix_class=mean_data.repeat([n_cluster_h,1])
    iscale_matrix=torch.linalg.inv(scale_matrix)
    center_clusters=center_clusters@iscale_matrix+shift_matrix_class
    #==========plot
    nx=len(av_lin_den)
    cluste_x=torch.arange(nx,device=device)+1
    if dim_data==2:
        data=data.cpu()
        label_data=label_data.cpu()
        center_clusters=center_clusters.cpu()
        plt.subplot(2,1,1)
        # plt.plot(data[:,0], data[:,1], '.b')
        for k_cluster in range(n_cluster_h):
            value, indices=((label_data==k_cluster)*1).sort(0)
            data_cluster=data[indices[value.argmax(0):],:]
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
            plt.plot(cluste_x, av_lin_den, '*r')
            plt.show()
        else:
            plt.plot(cluste_x, av_lin_den, '*r')
            plt.show()
    #==========================================================================
    return n_cluster_h, label_data, center_clusters

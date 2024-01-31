
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
from tqdm import tqdm


# ========================start of class=======================================
class Kalhor_LinearDensityIndex:
    def __init__(self, data, normalize=False):
        self.device1 = data.device
        # --------- input normalization
        if not normalize:
            self.data = data
        else:
            small_number = 1e-10
            m = data.mean(0)
            std = data.std(0) + small_number
            self.data = (data - m.reshape([1, -1]).repeat(data.shape[0], 1)) / std.reshape([1, -1]).repeat(data.shape[0], 1)
            print('data becomes normalized')
        # --------- target normalization
        self.big_number = 1e10
        self.n_data = data.shape[0]
        self.dim_data = data.shape[1]
        self.dis_matrix_input = torch.cdist(data, data, p=2).fill_diagonal_(self.big_number)
        self.pdist = torch.nn.PairwiseDistance(p=2)

    # (1. ldi_clustering method)==========================================
    def ldi_clustering(self, n_cluster_max,kmeans_repeat,dim_data):
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
        center_clusters=torch.zeros(n_cluster_max,dim_data, device=self.device1)
        div_lin=torch.zeros(dim_data,n_cluster_max, device=self.device1)
        bias_lin=torch.zeros(n_cluster_max,1, device=self.device1)
        mean_2=torch.zeros(2,dim_data, device=self.device1)
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
        for k_cluster in tqdm(range(n_cluster_max)):
            n_cluster=k_cluster+1
            #Find the worst cluster
            vmax, k_worst=div_score.max(0)
            #-----------divide the worst into two clusters
            value, indices=((label_data==k_worst)*1).sort()
            arg_worst=indices[value.argmax(0):]
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
            #===============================================
            center_clusters[k_worst,:]=mean_2[0,:]
            center_clusters[k_cluster,:]=mean_2[1,:]
            #

            label_data[arg_worst[arg_worst_down]]=k_cluster
            label_data[arg_worst[arg_worst_up]]=k_worst.int().item()

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
        return n_cluster_best, label_data_best, av_lin_den,sum_lin_den, center_clusters_best

    # (2. ldi_clustering_funct method)==========================================
    def ldi_clustering_func(self, data, n_cluster_max,kmeans_repeat):
        def get_param( label_data, k_cluster):
            value, indices=((label_data==k_cluster)*1).sort()
            inp=data[indices[value.argmax(0):],:]
            center=inp.mean(0)
            cov=torch.cov(inp.T)
            num_cluster=inp.shape[0]
            if dim_data>1:
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
        n_data=data.shape[0]
        dim_data=data.shape[1]
        lin_den=torch.zeros(n_cluster_max, device=self.device1)
        div_score=torch.zeros(n_cluster_max, device=self.device1)
        av_lin_den=torch.zeros(n_cluster_max, device=self.device1)
        num_clusters=torch.zeros(n_cluster_max, device=self.device1)
        center_clusters=torch.zeros(n_cluster_max,dim_data, device=self.device1)
        div_lin=torch.zeros(dim_data,n_cluster_max, device=self.device1)
        bias_lin=torch.zeros(n_cluster_max,1, device=self.device1)
        mean_2=torch.zeros(2,dim_data, device=self.device1)
        label_data=torch.zeros(n_data, device=self.device1 )
        # First Cluster
        n_cluster=1
        num_clusters[0]=n_data
        div_lin[:,0], bias_lin[0], div_score[0], lin_den[0], center_clusters[0,:], num_clusters[0]=get_param(label_data, 0)
        av_lin_den[0]=lin_den[0:n_cluster].mean()
        av_lin_den_max=av_lin_den[0]
        sum_lin_den=av_lin_den[0]
        label_data_best=label_data
        center_clusters_best=center_clusters[0:n_cluster,:]
        n_cluster_best=n_cluster
        for k_cluster in range(n_cluster_max):
            n_cluster=k_cluster+1
            #Find the worst cluster
            vmax, k_worst=div_score.max(0)
            #-----------divide the worst into two clusters
            value, indices=((label_data==k_worst)*1).sort()
            arg_worst=indices[value.argmax(0):]
            worst_data=data[arg_worst,:]
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
            #===============================================
            center_clusters[k_worst,:]=mean_2[0,:]
            center_clusters[k_cluster,:]=mean_2[1,:]
            #

            label_data[arg_worst[arg_worst_down]]=k_cluster
            label_data[arg_worst[arg_worst_up]]=k_worst.int().item()

            div_lin[:,k_worst], bias_lin[k_worst], div_score[k_worst], lin_den[k_worst], center_clusters[k_worst,:], num_clusters[k_worst]=get_param(label_data, k_worst)
            div_lin[:,k_cluster], bias_lin[k_cluster], div_score[k_cluster], lin_den[k_cluster], center_clusters[k_cluster,:], num_clusters[k_cluster]=get_param(label_data, k_cluster)
            av_lin_den[k_cluster]=lin_den[0:n_cluster].mean()
            if dim_data==1:
                av_lin_den[k_cluster]=lin_den[0:n_cluster].mean()/n_cluster
            if (av_lin_den[k_cluster] > av_lin_den_max):
                av_lin_den_max=av_lin_den[k_cluster]
                label_data_best=label_data.clone()
                center_clusters_best=center_clusters[0:n_cluster,:]
                sum_lin_den=av_lin_den[k_cluster]*n_cluster
                n_cluster_best=n_cluster
        return n_cluster_best, label_data_best, av_lin_den,sum_lin_den, center_clusters_best

    # (3. forward_feature_selection_ldi method)==========================================
    def backward_feature_selection_ldi(self, n_cluster_max,kmeans_repeat):
    # #===================================================================
        ranked_features = torch.arange(self.dim_data)
        n_cluster_ref, label_data_ref, av_lin_den_ref,sldi_ref, center_clusters_ref=self.ldi_clustering_func(self.data,n_cluster_max,kmeans_repeat)
        temp = torch.zeros(1, 1)
        worst_features = torch.zeros(1,0)
        num_clusters_f = torch.zeros(self.dim_data, 1, device=self.device1)
        sldi_ranked_features=torch.zeros(self.dim_data, 1, device=self.device1)
        sldi_ranked_features[0,0]=sldi_ref
        num_clusters_f[0,0]=n_cluster_ref
        for k_backward in range(self.dim_data-1):
            diff_min =self.dim_data*2
            for k_search in range(len(ranked_features)):
                ranked_features_search = torch.cat((ranked_features[0:k_search], ranked_features[k_search+1:]))
                n_cluster_best, label_data_best, av_lin_den,sldi, center_clusters_best=self.ldi_clustering_func(self.data[:,ranked_features_search],n_cluster_max,kmeans_repeat)

                if abs(n_cluster_best-n_cluster_ref) < diff_min:
                    num_clusters_best=n_cluster_best
                    diff_min = abs(n_cluster_best-n_cluster_ref)
                    sldi_best=sldi
                    ranked_features_search_best=ranked_features_search
                    worst_chosen_feature = ranked_features[k_search]

            temp[:, 0] = worst_chosen_feature.detach()
            worst_features = torch.cat((worst_features, temp), 1)
            ranked_features = ranked_features_search_best
            num_clusters_f[k_backward+1, 0]=num_clusters_best
            sldi_ranked_features[k_backward+1, 0]=sldi_best
        return sldi_ranked_features, worst_features, num_clusters_f

    # (4. assign_labels method)==========================================
    def assign_labels(self, n_cluster_max,kmeans_repeat, rare_data, rare_labels):
       n_cluster_best, label_data_best, av_lin_den,sldi, center_clusters_best=self.ldi_clustering(n_cluster_max,kmeans_repeat)
       print('n_cluster_best',n_cluster_best )
       print('label_data_best_max', label_data_best.max())
       label_data_best=label_data_best.int()
       dis=torch.cdist(center_clusters_best,rare_data, p=2)
       label_clusters=rare_labels[dis.argmin(1),:]
       assigned_label_data=label_clusters[label_data_best]
       return assigned_label_data
# =======================End of Class=================================

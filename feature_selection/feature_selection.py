# ===========Some notes about this example
import torch
import numpy as np



#[3] Start of the class Feature_Selection_Unsupervised =================================================================================================
class Feature_Selection_Unsueprvised:

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
   #====
     # (2. ldi_clustering method)==========================================
    def ldi_clustering_func(self, data, n_cluster_max,kmeans_repeat):
        dim_data=data.shape[1]
        def get_param( label_data, k_cluster):
            value, indices=((label_data==k_cluster)*1).sort()
            inp=data[indices[value.argmax(0):],:]
            center=inp.mean(0)
            # n_inp=inp.shape[0]
            # inp_tilda=inp.clone()
            # center_vec=center.repeat([n_inp,1])
            # inp_tilda-=center_vec
            # cov=(inp_tilda.T@inp_tilda)/(n_inp*1.-1)
            # print('covb', cov)
            cov=torch.cov(inp.T)
            # print('inp', inp)
            num_cluster=inp.shape[0]
            if  dim_data>1:
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
        center_clusters=torch.zeros(n_cluster_max, dim_data, device=self.device1)
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
        k_cluster=0
        stp=0
        while stp==0:

            #Find the worst cluster
            vmax, k_worst=div_score.max(0)
            #-----------divide the worst into two clusters
            value, indices=((label_data==k_worst)*1).sort()
            arg_worst=indices[value.argmax(0):]
            n_cluster=k_cluster+1
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
            #==============================================
            if (len(arg_worst_down)>1)&(len(arg_worst_up)>1):
                label_data[arg_worst[arg_worst_down]]=k_cluster
                label_data[arg_worst[arg_worst_up]]=k_worst.int().item()
                center_clusters[k_worst,:]=mean_2[0,:]
                center_clusters[k_cluster,:]=mean_2[1,:]
                div_lin[:,k_worst], bias_lin[k_worst], div_score[k_worst], lin_den[k_worst], center_clusters[k_worst,:], num_clusters[k_worst]=get_param(label_data, k_worst)
                div_lin[:,k_cluster], bias_lin[k_cluster], div_score[k_cluster], lin_den[k_cluster], center_clusters[k_cluster,:], num_clusters[k_cluster]=get_param(label_data, k_cluster)
                av_lin_den[k_cluster]=lin_den[0:n_cluster].mean()
                if  dim_data==1:
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

   # (3. forward_feature_selection_smi method)==========================================
    def forward_smi(self, smi_threshold):
        def smi_linear(data, target):
            dis_matrix_input = torch.cdist(data, data, p=2).fill_diagonal_(self.big_number)
            dis_matrix_target = torch.cdist(target, target, p=2)
            values, indices = torch.max(dis_matrix_target, 0)
            dis_target_max = values
            values, indices = torch.min(dis_matrix_target.fill_diagonal_(self.big_number), 0)
            dis_target_min = values
            pdist = torch.nn.PairwiseDistance(p=2)
            # ========================
            values, indices = torch.min(dis_matrix_input, 0)
            dis_target_star = pdist(target, target[indices, :])
            # ========================
            smi_data = (dis_target_max - dis_target_star) / (dis_target_max - dis_target_min)
            smi = torch.sum(smi_data) / data.shape[0]
            return smi
        tabel_smi=torch.zeros(self.dim_data,self.dim_data, device=self.device1)
        for k1 in range(self.dim_data):
            data=self.data[:,k1:k1+1]
            for k2 in range (self.dim_data):
                if k1!=k2:
                    target=self.data[:,k2:k2+1]
                    tabel_smi[k1,k2]=smi_linear(data, target)
        values, indices=tabel_smi.max(1)
        kworst=values.argmax()
        chosen_features=torch.arange(self.dim_data, device=self.device1)
        while (values[kworst]>smi_threshold)&(len(chosen_features)>1):
            chosen_features=torch.cat((chosen_features[0:kworst],chosen_features[kworst+1:]))
            tabel_smi=torch.cat((tabel_smi[:,0:kworst],tabel_smi[:,kworst+1:]),1)
            tabel_smi=torch.cat((tabel_smi[0:kworst,:],tabel_smi[kworst+1:,:]),0)
            values, indices=tabel_smi.max(1)
            kworst=values.argmax()
    # #===================================================================
        return chosen_features


    # (4. forward_feature_selection_ldi method)==========================================
    def forward_ldi(self, n_cluster_max,kmeans_repeat):
    # #===================================================================
        rest_features = torch.arange(self.dim_data).int()
        num_clusters_vec= torch.zeros(self.dim_data, 1, device=self.device1)
        best_features=torch.zeros(0, device=self.device1)
        num_clusters_max_max=0
        for k_forward in range(self.dim_data):
            num_clusters_max=0
            for k_search in range(len(rest_features)):
                best_features_search = torch.cat((best_features, rest_features[k_search:k_search+1]))
                rest_features1=torch.cat((rest_features[0:k_search], rest_features[k_search+1:]))
                best_features_search=best_features_search.int()
                data_chosen=self.data[:,best_features_search]
                n_cluster, label_data_best, av_lin_den,sldi, center_clusters_best=self.ldi_clustering_func(data_chosen,n_cluster_max,kmeans_repeat)
                if n_cluster>num_clusters_max:
                    num_clusters_max=n_cluster
                    best_features_temp=best_features_search.clone()
                    rest_features_temp=rest_features1.clone()
            best_features=best_features_temp.clone()
            rest_features=rest_features_temp.clone()
            num_clusters_vec[k_forward,:]=num_clusters_max
            if num_clusters_max>=num_clusters_max_max:
                num_clusters_max_max=num_clusters_max
                chosen_features=best_features.clone()

        return chosen_features, num_clusters_max_max

# ===End of class Feature_Selection_Unsupervised=================================================================================================





#[3] module_feature_selection_unsupervised
def module_feature_selection_unsupervised(data, normalize_do=True):
    n_cluster_max=100
    kmeans_repeat=20
    smi_threshold=0.8
    dim_data=data.shape[1]
    print('dim_data:', dim_data)
    dim_data=data.shape[1]
    print('n_data: ', data.shape[0])
    #[2]Data Feature Slection=============================
    instant_initial= Feature_Selection_Unsueprvised(data, normalize_do)
    ranked_features_best_smi=instant_initial.forward_smi(smi_threshold)
    print('ranked_features_best_smi',ranked_features_best_smi)

    data_smi=data[:,ranked_features_best_smi]
    instant_initial= Feature_Selection_Unsueprvised(data_smi, normalize_do)
    ranked_features_best_ldi, num_clusters_max_max=instant_initial.forward_ldi(n_cluster_max,kmeans_repeat)
    ranked_features_best, arg_sort=ranked_features_best_smi[ranked_features_best_ldi].sort()
    print('ranked_features_best',ranked_features_best)
    print('num_clusters_max_max',num_clusters_max_max)
    dim_data_best=len(ranked_features_best)
    print('dim_data_best: ',dim_data_best)
    #================================================
    data_best=data[:,ranked_features_best]
    return ranked_features_best, data_best

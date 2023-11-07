# ===========Some notes about this example
import torch



class Data_Scoring_Unsupervised:
    def __init__(self, data, black_hole):
        self.device1 = data.device
        # --------- target normalization
        self.big_number = 1e10
        self.n_data = data.shape[0]
        self.dim_data = data.shape[1]
        self.dis_matrix_input = torch.cdist(data, data, p=2).fill_diagonal_(self.big_number)
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.black_hole=black_hole
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
        def balck_hole(n):
            if self.black_hole:
                n_saturated=torch.tanh(torch.tensor(2*n/self.n_data))
            else:
                n_saturated=n
            return n_saturated

        def get_param( label_data, k_cluster):
            value, indices=((label_data==k_cluster)*1).sort()
            inp=self.data[indices[value.argmax(0):],:]
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
            if self.dim_data>1:
                Eig_vec, Eig_val, V=cov.svd()
                lin_den=num_cluster/Eig_val[0]
                div_score=balck_hole(num_cluster)*Eig_val[0]
                div_lin=Eig_vec[:,0]
                bias_lin=-center@div_lin
            else:
                lin_den=num_cluster/cov
                div_score=balck_hole(num_cluster)*cov
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

# ===End of class Feature_Selection_Regression=================================================================================================


#==========================================================================================================================
#[3] module_data_scoring_unsupervised

def module_data_scoring_unsupervised(data,n_max_clusters, black_hole=False, normalize_do=False):
    device = data.device
    print('data', data.shape)
    print('n_max_clusters', n_max_clusters)
    kmeans_repeat=100
    dim_data=data.shape[1]
    n_data=data.shape[0]
    if normalize_do:
        instant_intial = Data_Scoring_Unsupervised(data, black_hole)
        mean_data, scale_matrix=instant_intial.get_normalization_matrix()
        shift_matrix=mean_data.repeat([n_data,1])
        data_norm=(data-shift_matrix)@scale_matrix
    else:
        mean_data=torch.zeros(dim_data, device=device)
        scale_matrix=torch.eye(dim_data, device=device)
        shift_matrix=mean_data.repeat([n_data,1])
        data_norm=(data-shift_matrix)@scale_matrix

    #=============================================
    instant = Data_Scoring_Unsupervised(data_norm, black_hole)
    n_cluster_h, label_data, av_lin_den,sum_lin_den, center_clusters = instant.ldi_clustering(n_max_clusters, kmeans_repeat)
    print('the predicted number of clusters is:',n_cluster_h )
    shift_matrix_class=mean_data.repeat([n_cluster_h,1])
    iscale_matrix=torch.linalg.inv(scale_matrix)
    center_clusters=center_clusters@iscale_matrix+shift_matrix_class
    #==========================================================================
    distance=torch.cdist(data,center_clusters,p=2)
    values, indices= distance.sort(1)
    if n_cluster_h>1:
        data_scores=1-values[:,0]/values[:,1]
    else:
        data_scores=1-values[:,0]/values[:,-1]

    return data_scores

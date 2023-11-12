#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:11:36 2023
This code was written by Dr. Ahmad Kalhor (ad.kalhor@gmail.com)in pytorch framework
It includes "Kalhor_SmoothnessIndex" class with 29 methods.
"""
import torch
import numpy as np
# ===========Some notes about the class
# All methods of the class are developed from the concept of Smoothness Index(SmI).
# SmI is a normalized supervised data complexity measure which is utilised to analyze and design AI-models in Regression problems.
# Each method actually measures a variant  smoothness of the input data points with target data points.
# To illustrate  how one can use this class, an illustrative example is provided in an executable file.

# ==========Some applications of using "Kalhor_SeparationIndex" class:
# 1-To evaluate different datasets (datasets with lower SmI (SmI--->0) are more challenging than those with high SmI (SmI--->1))
# 2-To evaluate different layers of a shallow/deep classification model(it is expected that in an appropriate learned model, SmI increases (in average) to one layer by layer)
# 3-To rank different models (models which provide higher SIs are more appropriate)
# 4-To check the generalization of a model( models with higher cross-SmI have more generalization)
# 5-To rank different features and select the best subset in a classification problem(features which makes higher SmI are better)
# 6-To clean datasets or models which are poisoned/attacked by backdoors/trojans/biases (SmI can detect such issues and remove them)
# 7-To Compress a classification model (layers and nodes which can not increase the SmI are removed)
# 8-To learn a model in a forward layer-wise manner (using a ranking loss (to maximize the SmI) at each layer, one can get better results than conventional backpropagation methods)
# 9-To find some key layers and nodes where a model makes some branches or being fused by other models in order to maximize the SmI.
# 10-To determine a trustworthy/confidence/guarantee for the predictions of the model (High order (or soft order) SmIs are utilized) .
# 11-To remove risky data points and augment new effective data points for better training and generalization.




# ========================start of class=======================================
class Kalhor_SmoothnessIndex:
    def __init__(self, inp, target, inp_normalize=False, target_normalize=False):
        self.device1 = inp.device
        target.to(self.device1)
        # --------- input normalization
        self.inp_normalize=inp_normalize
        self.target_normalize=target_normalize
        if not inp_normalize:
            self.inp = inp
        else:
            small_number = 1e-10
            m = inp.mean(0)
            std = inp.std(0) + small_number
            self.inp = (inp - m.reshape([1, -1]).repeat(inp.shape[0], 1)) / std.reshape([1, -1]).repeat(inp.shape[0], 1)
            self.inp_std=std
            self.inp_m=m
            print('input data becomes normalized')
        # --------- target normalization
        if not target_normalize:
            self.target = target
        else:
            small_number = 1e-10
            m = target.mean(0)
            std = target.std(0) + small_number
            self.target = (target - m.reshape([1, -1]).repeat(target.shape[0], 1)) / std.reshape([1, -1]).repeat(
                target.shape[0], 1)
            self.target_std=std
            self.target_m=m
            print('target data becomes normalized')

        self.big_number = 1e10
        self.n_data = self.inp.shape[0]
        self.dim_input = self.inp.shape[1]
        self.dim_target = self.target.shape[1]
        self.dis_matrix_input = torch.cdist(self.inp, self.inp, p=2).fill_diagonal_(self.big_number)
        dis_matrix_target = torch.cdist(self.target, self.target, p=2)
        values, indices = torch.max(dis_matrix_target, 0)
        self.dis_target_max = values
        self.dis_target_mean = torch.mean(dis_matrix_target, 0)
        values, indices = torch.min(dis_matrix_target.fill_diagonal_(self.big_number), 0)
        self.dis_target_min = values
        self.dis_matrix_target = dis_matrix_target.fill_diagonal_(self.big_number)
        self.pdist = torch.nn.PairwiseDistance(p=2)

    # (1. smi_linear method)==========================================
    def smi_linear(self):
        # ========================
        values, indices = torch.min(self.dis_matrix_input, 0)
        dis_target_star = self.pdist(self.target, self.target[indices, :])
        # ========================
        smi_data = (self.dis_target_max - dis_target_star) / (self.dis_target_max - self.dis_target_min)
        smi = torch.sum(smi_data) / self.n_data
        return smi

    # (2. smi_linear_data method)==========================================
    def smi_linear_data(self):
        values, indices = torch.min(self.dis_matrix_input, 0)
        dis_target_star = self.pdist(self.target, self.target[indices, :])
        # ========================
        smi_data = (self.dis_target_max - dis_target_star) / (self.dis_target_max - self.dis_target_min)
        return smi_data

    # (3. smi_exp method)==========================================
    def smi_exp(self, gama):
        values, indices = torch.min(self.dis_matrix_input, 0)
        dis_target_star = self.pdist(self.target, self.target[indices, :])
        # ========================
        smi_data = torch.exp(-gama * (dis_target_star - self.dis_target_min) / self.dis_target_mean)
        smi = torch.sum(smi_data) / self.n_data
        return smi

    # (4. smi_exp_data method)==========================================
    def smi_exp_data(self, gama):
        values, indices = torch.min(self.dis_matrix_input, 0)
        dis_target_star = self.pdist(self.target, self.target[indices, :])
        # ========================
        smi_data = torch.exp(-gama * (dis_target_star - self.dis_target_min) / self.dis_target_mean)
        return smi_data

    # (5. high_order_smi_linear method)==========================================
    def high_order_smi_linear(self, order):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_max_2d = self.dis_target_max.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = (dis_max_2d - dis_star_2d) / (dis_max_2d - dis_min_2d)
        high_si_data, indices = smi_data_2d.min(1)
        high_si = torch.sum(high_si_data) / self.n_data
        return high_si

    # (6. high_order_smi_linear_data method)==========================================
    def high_order_smi_linear_data(self, order):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_max_2d = self.dis_target_max.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = (dis_max_2d - dis_star_2d) / (dis_max_2d - dis_min_2d)
        high_si_data, indices = smi_data_2d.min(1)
        return high_si_data

    # (7. high_order_smi_exp method)==========================================
    def high_order_smi_exp(self, order, gama):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_mean_2d = self.dis_target_mean.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = torch.exp(-gama * (dis_star_2d - dis_min_2d) / dis_mean_2d)
        high_si_data, indices = smi_data_2d.min(1)
        high_si = torch.sum(high_si_data) / self.n_data
        return high_si

    # (8. high_order_smi_exp_data method)==========================================
    def high_order_smi_exp_data(self, order, gama):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_mean_2d = self.dis_target_mean.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = torch.exp(-gama * (dis_star_2d - dis_min_2d) / dis_mean_2d)
        high_si_data, indices = smi_data_2d.min(1)
        return high_si_data

    # (9. anti_smi_linear method)==========================================
    def anti_smi_linear(self, order):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_max_2d = self.dis_target_max.reshape([-1, 1]).repeat([1, order])
        anti_smi_data_2d = (dis_star_2d - dis_min_2d) / (dis_max_2d - dis_min_2d)
        anti_smi_data, indices = anti_smi_data_2d.min(1)
        anti_smi = torch.sum(anti_smi_data) / self.n_data
        return anti_smi

    # (10. anti_smi_linear_data method)==========================================
    def anti_smi_linear_data(self, order):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_max_2d = self.dis_target_max.reshape([-1, 1]).repeat([1, order])
        anti_smi_data_2d = (dis_star_2d - dis_min_2d) / (dis_max_2d - dis_min_2d)
        anti_smi_data, indices = anti_smi_data_2d.min(1)
        return anti_smi_data

    # (11. anti_smi_exp method)==========================================
    def anti_smi_exp(self, order, gama):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_mean_2d = self.dis_target_mean.reshape([-1, 1]).repeat([1, order])
        anti_smi_data_2d = 1 - torch.exp(-gama * (dis_star_2d - dis_min_2d) / dis_mean_2d)
        anti_smi_data, indices = anti_smi_data_2d.min(1)
        anti_smi = torch.sum(anti_smi_data) / self.n_data
        return anti_smi

    # (12. anti_smi_exp_data method)==========================================
    def anti_smi_exp_data(self, order, gama):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_mean_2d = self.dis_target_mean.reshape([-1, 1]).repeat([1, order])
        anti_smi_data_2d = 1 - torch.exp(-gama * (dis_star_2d - dis_min_2d) / dis_mean_2d)
        anti_smi_data, indices = anti_smi_data_2d.min(1)
        return anti_smi_data

    # (13. soft_order_smi_linear method)==========================================
    def soft_order_smi_linear(self, order):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_max_2d = self.dis_target_max.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = (dis_max_2d - dis_star_2d) / (dis_max_2d - dis_min_2d)
        soft_smi_data = smi_data_2d.mean(1)
        soft_smi = torch.sum(soft_smi_data) / self.n_data
        return soft_smi

    # (14. soft_order_smi_linear_data method)==========================================
    def soft_order_smi_linear_data(self, order):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_max_2d = self.dis_target_max.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = (dis_max_2d - dis_star_2d) / (dis_max_2d - dis_min_2d)
        soft_smi_data = smi_data_2d.mean(1)
        return soft_smi_data

    # (15. soft_order_smi_exp method)==========================================
    def soft_order_smi_exp(self, order, gama):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_mean_2d = self.dis_target_mean.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = torch.exp(-gama * (dis_star_2d - dis_min_2d) / dis_mean_2d)
        soft_smi_data = smi_data_2d.mean(1)
        soft_smi = torch.sum(soft_smi_data) / self.n_data
        return soft_smi

    # (16. soft_order_smi_exp_data method)==========================================
    def soft_order_smi_exp_data(self, order, gama):
        # ========================
        values, indices_star = torch.sort(self.dis_matrix_input, 1)
        # ========================
        values, indices_sort = torch.sort(self.dis_matrix_target, 1)
        # ========================
        target_3d = self.target.reshape([-1, 1, self.dim_target]).repeat([1, order, 1])
        taregt_star_3d = self.target[indices_star[:, 0:order], :]
        taregt_min_3d = self.target[indices_sort[:, 0:order], :]
        dis_star_2d = self.pdist(target_3d, taregt_star_3d)
        dis_min_2d = self.pdist(target_3d, taregt_min_3d)
        dis_mean_2d = self.dis_target_mean.reshape([-1, 1]).repeat([1, order])
        smi_data_2d = torch.exp(-gama * (dis_star_2d - dis_min_2d) / dis_mean_2d)
        soft_smi_data = smi_data_2d.mean(1)
        return soft_smi_data

    # (17. cross_smi_linear method)==========================================
    def cross_smi_linear(self, inp_test, target_test):

        if self.inp_normalize:
            inp_test = (inp_test - self.inp_m.reshape([1, -1]).repeat(inp_test.shape[0], 1)) / self.inp_std.reshape([1, -1]).repeat(inp_test.shape[0], 1)
        # --------- target normalization
        if self.target_normalize:
            target_test = (target_test - self.target_m.reshape([1, -1]).repeat(target_test.shape[0], 1)) / self.target_std.reshape([1, -1]).repeat(
                target_test.shape[0], 1)

        n_test, n_data = inp_test.shape
        dis_matrix = torch.cdist(inp_test, self.inp, p=2)
        values, indices = torch.min(dis_matrix, 1)
        dis_target_star = self.pdist(target_test, self.target[indices, :])
        # ========================
        dis_matrix = torch.cdist(target_test, self.target, p=2)
        #
        dis_target_min, indices = torch.min(dis_matrix, 1)
        #
        dis_target_max, indices = torch.max(dis_matrix, 1)
        # ========================
        cr_smi_data = (dis_target_max - dis_target_star) / (dis_target_max - dis_target_min)
        cross_smi = torch.sum(cr_smi_data) / n_test
        return cross_smi

    # (18. cross_smi_linear_data method)==========================================
    def cross_smi_linear_data(self, inp_test, target_test):
        if self.inp_normalize:
            inp_test = (inp_test - self.inp_m.reshape([1, -1]).repeat(inp_test.shape[0], 1)) / self.inp_std.reshape([1, -1]).repeat(inp_test.shape[0], 1)
        # --------- target normalization
        if self.target_normalize:
            target_test = (target_test - self.target_m.reshape([1, -1]).repeat(target_test.shape[0], 1)) / self.target_std.reshape([1, -1]).repeat(
                target_test.shape[0], 1)


        dis_matrix = torch.cdist(inp_test, self.inp, p=2)
        values, indices = torch.min(dis_matrix, 1)
        dis_target_star = self.pdist(target_test, self.target[indices, :])
        # ========================
        dis_matrix = torch.cdist(target_test, self.target, p=2)
        #
        dis_target_min, indices = torch.min(dis_matrix, 1)

        #
        dis_target_max, indices = torch.max(dis_matrix, 1)
        # ========================
        cross_smi_data = (dis_target_max - dis_target_star) / (dis_target_max - dis_target_min)
        return cross_smi_data

    # (19. cross_smi_exp method)==========================================
    def cross_smi_exp(self, inp_test, target_test, gama):

        if self.inp_normalize:
            inp_test = (inp_test - self.inp_m.reshape([1, -1]).repeat(inp_test.shape[0], 1)) / self.inp_std.reshape([1, -1]).repeat(inp_test.shape[0], 1)
        # --------- target normalization
        if self.target_normalize:
            target_test = (target_test - self.target_m.reshape([1, -1]).repeat(target_test.shape[0], 1)) / self.target_std.reshape([1, -1]).repeat(
                target_test.shape[0], 1)

        n_test, n_data = inp_test.shape
        dis_matrix = torch.cdist(inp_test, self.inp, p=2)
        values, indices = torch.min(dis_matrix, 1)
        dis_target_star = self.pdist(target_test, self.target[indices, :])
        # ========================
        dis_matrix = torch.cdist(target_test, self.target, p=2)
        #
        dis_target_min, indices = torch.min(dis_matrix, 1)
        #
        dis_target_mean = torch.mean(dis_matrix, 1)

        # ========================
        cr_smi_data = torch.exp(-gama * (dis_target_star - dis_target_min) / dis_target_mean)
        cross_smi = torch.sum(cr_smi_data) / n_test
        return cross_smi

    # (20. cross_smi_exp_data method)==========================================
    def cross_smi_exp_data(self, inp_test, target_test, gama):
        if self.inp_normalize:
            inp_test = (inp_test - self.inp_m.reshape([1, -1]).repeat(inp_test.shape[0], 1)) / self.inp_std.reshape([1, -1]).repeat(inp_test.shape[0], 1)
        # --------- target normalization
        if self.target_normalize:
            target_test = (target_test - self.target_m.reshape([1, -1]).repeat(target_test.shape[0], 1)) / self.target_std.reshape([1, -1]).repeat(
                target_test.shape[0], 1)
        dis_matrix = torch.cdist(inp_test, self.inp, p=2)
        values, indices = torch.min(dis_matrix, 1)
        dis_target_star = self.pdist(target_test, self.target[indices, :])
        # ========================
        dis_matrix = torch.cdist(target_test, self.target, p=2)
        dis_target_min, indices = torch.min(dis_matrix, 1)
        #
        dis_target_mean = torch.mean(dis_matrix, 1)
        # ========================
        cross_smi_data = torch.exp(-gama * (dis_target_star - dis_target_min) / dis_target_mean)
        return cross_smi_data

    # (21. triplet_local_smi method)
    def triplet_local_smi(self, n_neighb):

        values, indices_sort = torch.sort(self.dis_matrix_input, 1)
        #=====method1 (Fast)
        # column_random=torch.randperm(n_neighb, device=self.device1)
        # r2 = indices_sort[:,column_random[0]]
        # r3 = indices_sort[:,column_random[1]]
        #=====method2 (with more randomness)
        r2 = torch.zeros(self.n_data, device=self.device1).int()
        r3 = torch.zeros(self.n_data, device=self.device1).int()
        n=self.n_data;x = torch.zeros(n, n);perm = torch.stack([torch.randperm(n_neighb) for _ in range(len(x))])
        for j in range(self.n_data):
            r2[j]=indices_sort[j,perm[j,0]]
            r3[j]=indices_sort[j,perm[j,1]]

        #=====End of method2

        diff_target = (torch.pairwise_distance(self.target, self.target[r2, :]) - torch.pairwise_distance(self.target,
                                                                                                          self.target[
                                                                                                          r3,
                                                                                                          :])).sign()
        diff_inp = (torch.pairwise_distance(self.inp, self.inp[r2, :]) - torch.pairwise_distance(self.inp, self.inp[r3,
                                                                                                           :])).sign()
        #
        l_smi = ((diff_target == diff_inp) * 1.0).mean()
        #
        sort_target, arg_target = torch.sort(diff_target)
        arg_part1 = arg_target[0:torch.argmax(sort_target)]
        arg_part2 = arg_target[torch.argmax(sort_target):]
        # data_ancher=self.inp
        inp_positive = self.inp[torch.cat((r2[arg_part1], r3[arg_part2])), :]
        inp_negative = self.inp[torch.cat((r3[arg_part1], r2[arg_part2])), :]
        return l_smi, self.inp, inp_positive, inp_negative
    # (22. triplet_global_smi method)
    def triplet_global_smi(self):

        r2 = torch.randperm(self.n_data, device=self.device1)
        r3 = torch.randperm(self.n_data, device=self.device1)
        #
        diff_target = (torch.pairwise_distance(self.target, self.target[r2, :]) - torch.pairwise_distance(self.target,
                                                                                                          self.target[
                                                                                                          r3,
                                                                                                          :])).sign()
        diff_inp = (torch.pairwise_distance(self.inp, self.inp[r2, :]) - torch.pairwise_distance(self.inp, self.inp[r3,
                                                                                                           :])).sign()
        #
        g_smi = ((diff_target == diff_inp) * 1.0).mean()
        #
        sort_target, arg_target = torch.sort(diff_target)
        arg_part1 = arg_target[0:torch.argmax(sort_target)]
        arg_part2 = arg_target[torch.argmax(sort_target):]
        # data_ancher=self.inp
        inp_positive = self.inp[torch.cat((r2[arg_part1], r3[arg_part2])), :]
        inp_negative = self.inp[torch.cat((r3[arg_part1], r2[arg_part2])), :]
        return g_smi, self.inp, inp_positive, inp_negative
    # (23. data_dividing_smi method)==========================================
    def data_dividing_smi(self, smi_data):
        n_data_easy = torch.sum(smi_data)
        n_data_difficult = self.n_data - n_data_easy.detach()
        si_sort, arg_sort = torch.sort(smi_data, 0)
        inp_difficult = self.data[arg_sort[0:n_data_difficult, 0], :]
        target_difficult = self.label[arg_sort[0:n_data_difficult, 0]]
        inp_easy = self.inp[arg_sort[n_data_difficult:, 0], :]
        target_easy = self.target[arg_sort[n_data_difficult:, 0], 0]
        return inp_difficult, target_difficult, inp_easy, target_easy

    # (24. signle_input_smi_linear method)==========================================
    def signle_input_smi_linear(self):
        data_3d = self.inp.reshape([-1, 1, self.dim_input]).repeat([1, self.n_data, 1])
        tr_data_3d = data_3d.transpose(0, 1)
        distanc_3d = data_3d ** 2
        tr_distanc_3d = distanc_3d.transpose(0, 1)
        eye_3d = torch.eye(self.n_data, device=self.device1).reshape([self.n_data, self.n_data, 1]).repeat(
            [1, 1, self.dim_input])
        dis_matrix_features = eye_3d * self.big_number + distanc_3d + tr_distanc_3d - 2 * (data_3d * tr_data_3d)
        values, indices = dis_matrix_features.min(1)
        #
        target_star = self.target[indices]
        target_star = target_star[:, :, 0]
        dis_star_2d = torch.abs(self.target - target_star)
        dis_min_2d = self.dis_target_min.reshape([-1, 1]).repeat([1, self.dim_input])
        dis_max_2d = self.dis_target_max.reshape([-1, 1]).repeat([1, self.dim_input])
        signle_feature_smi = (dis_max_2d - dis_star_2d) / (dis_max_2d - dis_min_2d)
        signle_feature_smi = signle_feature_smi.sum(0) / self.n_data

        return signle_feature_smi

    # (25. signle_input_smi_exp method)==========================================
    def signle_input_smi_exp(self, gama):
        data_3d = self.inp.reshape([-1, 1, self.dim_input]).repeat([1, self.n_data, 1])
        tr_data_3d = data_3d.transpose(0, 1)
        distanc_3d = data_3d ** 2
        tr_distanc_3d = distanc_3d.transpose(0, 1)
        eye_3d = torch.eye(self.n_data, device=self.device1).reshape([self.n_data, self.n_data, 1]).repeat(
            [1, 1, self.dim_input])
        dis_matrix_features = eye_3d * self.big_number + distanc_3d + tr_distanc_3d - 2 * (data_3d * tr_data_3d)
        values, indices = dis_matrix_features.min(1)
        #
        target_star = self.target[indices]
        target_star = target_star[:, :, 0]
        dis_star_2d = torch.abs(self.target - target_star)
        dis_min_2d = self.dis_target_min.reshape([-1, 1]).repeat([1, self.dim_input])
        dis_mean_2d = self.dis_target_mean.reshape([-1, 1]).repeat([1, self.dim_input])
        signle_feature_smi = torch.exp(-gama * (dis_star_2d - dis_min_2d) / dis_mean_2d)
        signle_feature_smi = signle_feature_smi.sum(0) / self.n_data
        return signle_feature_smi

    # (26. forward_input_ranking_smi_linear method)==========================================
    def forward_input_ranking_smi_linear(self):
        ####
        ranked_features = torch.zeros(1, 0)
        temp = torch.zeros(1, 1)
        rest_features = torch.arange(self.dim_input)
        smi_ranked_features = torch.zeros(self.dim_input, 1, device=(self.device1))
        # data_3d = self.inp.reshape([-1, 1, self.dim_input]).repeat([1, self.n_data, 1])
        # tr_data_3d = data_3d.transpose(0, 1)
        # distanc_3d = data_3d ** 2
        # tr_distanc_3d = distanc_3d.transpose(0, 1)
        # eye_3d = torch.eye(self.n_data, device=self.device1).reshape([self.n_data, self.n_data, 1]).repeat(
        #     [1, 1, self.dim_input])
        # dis_matrix_features = eye_3d * self.big_number + distanc_3d + tr_distanc_3d - 2 * (data_3d * tr_data_3d)
        for k_forward in range(self.dim_input):
            smi_max = 0
            for k_search in range(len(rest_features)):
                ranked_features_search = np.append(ranked_features, rest_features[k_search])
                # dis_features_search = torch.sum(dis_matrix_features[:, :, ranked_features_search], 2)
                inp1=self.inp[:,ranked_features_search]
                dis_features_search=torch.cdist(inp1, inp1, p=2).fill_diagonal_(self.big_number)
                values, indices = torch.min(dis_features_search, 1)

                dif_target = self.target - self.target[indices, :]
                dis_target_star = torch.sum(dif_target * dif_target, 1) ** 0.5
                # ========================
                smi_data = (self.dis_target_max - dis_target_star) / (self.dis_target_max - self.dis_target_min)
                smi = torch.sum(smi_data).detach() / self.n_data
                if smi > smi_max:
                    smi_max = smi
                    chosen_feature = rest_features[k_search]
                    k_search_chosen = k_search
            temp[:, 0] = chosen_feature.detach()
            ranked_features = torch.cat((ranked_features, temp), 1)
            rest_features = torch.cat([rest_features[:k_search_chosen], rest_features[k_search_chosen + 1:]])
            smi_ranked_features[k_forward, 0] = smi_max
        return smi_ranked_features, ranked_features

    # (27. forward_input_ranking_smi_exp method)==========================================
    def forward_input_ranking_smi_exp(self, gama):
        ####
        ranked_features = torch.zeros(1, 0)
        temp = torch.zeros(1, 1)
        rest_features = torch.arange(self.dim_input)
        smi_ranked_features = torch.zeros(self.dim_input, 1, device=(self.device1))
        # data_3d = self.inp.reshape([-1, 1, self.dim_input]).repeat([1, self.n_data, 1])
        # tr_data_3d = data_3d.transpose(0, 1)
        # distanc_3d = data_3d ** 2
        # tr_distanc_3d = distanc_3d.transpose(0, 1)
        # eye_3d = torch.eye(self.n_data, device=self.device1).reshape([self.n_data, self.n_data, 1]).repeat(
        #     [1, 1, self.dim_input])
        # dis_matrix_features = eye_3d * self.big_number + distanc_3d + tr_distanc_3d - 2 * (data_3d * tr_data_3d)

        for k_forward in range(self.dim_input):
            smi_max = 0
            for k_search in range(len(rest_features)):
                ranked_features_search = np.append(ranked_features, rest_features[k_search])
                # dis_features_search = torch.sum(dis_matrix_features[:, :, ranked_features_search], 2)
                inp1=self.inp[:,ranked_features_search]
                dis_features_search=torch.cdist(inp1, inp1, p=2).fill_diagonal_(self.big_number)


                values, indices = torch.min(dis_features_search, 1)
                dif_target = self.target - self.target[indices, :]
                dis_target_star = torch.sum(dif_target * dif_target, 1) ** 0.5
                # ========================
                smi_data = torch.exp(-gama * (dis_target_star - self.dis_target_min) / self.dis_target_mean)
                smi = torch.sum(smi_data).detach() / self.n_data
                if smi > smi_max:
                    smi_max = smi
                    chosen_feature = rest_features[k_search]
                    k_search_chosen = k_search
            temp[:, 0] = chosen_feature.detach()
            ranked_features = torch.cat((ranked_features, temp), 1)
            rest_features = torch.cat([rest_features[:k_search_chosen], rest_features[k_search_chosen + 1:]])
            smi_ranked_features[k_forward, 0] = smi_max
        return smi_ranked_features, ranked_features

    # (28. get_best_inputs_forward_smi_linear method)==========================================
    def get_best_inputs_forward_smi_linear(self):

        ####
        ranked_features = torch.zeros(1, 0)
        temp = torch.zeros(1, 1)
        rest_features = torch.arange(self.dim_input)
        smi_ranked_features = torch.zeros(self.dim_input, 1, device=(self.device1))
        # data_3d = self.inp.reshape([-1, 1, self.dim_input]).repeat([1, self.n_data, 1])
        # tr_data_3d = data_3d.transpose(0, 1)
        # distanc_3d = data_3d ** 2
        # tr_distanc_3d = distanc_3d.transpose(0, 1)
        # eye_3d = torch.eye(self.n_data, device=self.device1).reshape([self.n_data, self.n_data, 1]).repeat(
        #     [1, 1, self.dim_input])
        # dis_matrix_features = eye_3d * self.big_number + distanc_3d + tr_distanc_3d - 2 * (data_3d * tr_data_3d)
        for k_forward in range(self.dim_input):
            smi_max = 0
            for k_search in range(len(rest_features)):
                ranked_features_search = np.append(ranked_features, rest_features[k_search])
                # dis_features_search = torch.sum(dis_matrix_features[:, :, ranked_features_search], 2)
                inp1=self.inp[:,ranked_features_search]
                dis_features_search=torch.cdist(inp1, inp1, p=2).fill_diagonal_(self.big_number)
                values, indices = torch.min(dis_features_search, 1)

                dif_target = self.target - self.target[indices, :]
                dis_target_star = torch.sum(dif_target * dif_target, 1) ** 0.5
                # ========================
                smi_data = (self.dis_target_max - dis_target_star) / (self.dis_target_max - self.dis_target_min)
                smi = torch.sum(smi_data).detach() / self.n_data
                if smi > smi_max:
                    smi_max = smi
                    chosen_feature = rest_features[k_search]
                    k_search_chosen = k_search
            temp[:, 0] = chosen_feature.detach()
            ranked_features = torch.cat((ranked_features, temp), 1)
            rest_features = torch.cat([rest_features[:k_search_chosen], rest_features[k_search_chosen + 1:]])
            smi_ranked_features[k_forward, 0] = smi_max
        smi_max, arg_max = torch.max(smi_ranked_features, 0)
        ranked_inputs_best = ranked_features[:, 0:arg_max + 1]
        ranked_inputs_best = ranked_inputs_best.t().to(torch.long)

        return self.inp[:, ranked_inputs_best[:, 0]], ranked_inputs_best

    # (29. get_best_inputs_forward_smi_linear method)==========================================
    def get_best_inputs_forward_smi_exp(self, gama):

        ranked_features = torch.zeros(1, 0)
        temp = torch.zeros(1, 1)
        rest_features = torch.arange(self.dim_input)
        smi_ranked_features = torch.zeros(self.dim_input, 1, device=(self.device1))

        # data_3d = self.inp.reshape([-1, 1, self.dim_input]).repeat([1, self.n_data, 1])
        # tr_data_3d = data_3d.transpose(0, 1)
        # distanc_3d = data_3d ** 2
        # tr_distanc_3d = distanc_3d.transpose(0, 1)
        # eye_3d = torch.eye(self.n_data, device=self.device1).reshape([self.n_data, self.n_data, 1]).repeat(
        #     [1, 1, self.dim_input])
        # dis_matrix_features = eye_3d * self.big_number + distanc_3d + tr_distanc_3d - 2 * (data_3d * tr_data_3d)

        for k_forward in range(self.dim_input):
            smi_max = 0
            for k_search in range(len(rest_features)):
                ranked_features_search = np.append(ranked_features, rest_features[k_search])
                # dis_features_search = torch.sum(dis_matrix_features[:, :, ranked_features_search], 2)
                inp1=self.inp[:,ranked_features_search]
                dis_features_search=torch.cdist(inp1, inp1, p=2).fill_diagonal_(self.big_number)
                values, indices = torch.min(dis_features_search, 1)

                dif_target = self.target - self.target[indices, :]
                dis_target_star = torch.sum(dif_target * dif_target, 1) ** 0.5
                # ========================
                smi_data = torch.exp(-gama * (dis_target_star - self.dis_target_min) / self.dis_target_mean)
                smi = torch.sum(smi_data).detach() / self.n_data
                if smi > smi_max:
                    smi_max = smi
                    chosen_feature = rest_features[k_search]
                    k_search_chosen = k_search
            temp[:, 0] = chosen_feature.detach()
            ranked_features = torch.cat((ranked_features, temp), 1)
            rest_features = torch.cat([rest_features[:k_search_chosen], rest_features[k_search_chosen + 1:]])
            smi_ranked_features[k_forward, 0] = smi_max
        # ===================================================
        smi_max, arg_max = torch.max(smi_ranked_features, 0)
        ranked_features_best = ranked_features[:, 0:arg_max + 1]
        ranked_features_best = ranked_features_best.t().to(torch.long)
        return self.inp[:, ranked_features_best[:, 0]], ranked_features_best
# =======================End of Class=================================

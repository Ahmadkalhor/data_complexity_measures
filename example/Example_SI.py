"""
Created on Mon Jun 26 17:11:36 2023
This code was written by Dr. Ahmad Kalhor (ad.kalhor@gmail.com)
"Kalhor_SeparationIndex" is a class (written in pytorch framework) which includes some methods to analyze classification datasets and models.
"""

from .SeparationIndex import Kalhor_SeparationIndex
# ===========Some notes abo ut this example


import torch
import numpy as np
import matplotlib.pyplot as plt


# a toy example function which generate a two-dimensional classification dataset
def toy_example(n_data, n_feature, n_class, n_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.zeros(n_data, n_feature, device=device)
    label = torch.zeros(n_data, 1, device=device)
    data_test = torch.zeros(n_test, n_feature, device=device)
    label_test = torch.zeros(n_test, 1, device=device)
    # ====================================================train data
    # class1
    label[0:3, 0] = 1
    data[0, 0] = 1
    data[1, 0] = 0.8
    data[2, 0] = 1.2
    data[0, 1] = 2
    data[1, 1] = 1.7
    data[2, 1] = 1.7
    # class2
    label[3:7, 0] = 2
    data[3, 0] = 2
    data[4, 0] = 1.8
    data[5, 0] = 2.2
    data[6, 0] = 1
    data[3, 1] = 1
    data[4, 1] = 0.8
    data[5, 1] = 0.8
    data[6, 1] = 1
    # class3
    label[7:11, 0] = 3
    data[7, 0] = 3
    data[8, 0] = 2.8
    data[9, 0] = 3.2
    data[10, 0] = 2.5
    data[7, 1] = 2
    data[8, 1] = 1.8
    data[9, 1] = 1.8
    data[10, 1] = 0.8
    # ====================================================test data
    # class1
    label_test[0:2, 0] = 1
    data_test[0, 0] = 1.2
    data_test[1, 0] = 2
    data_test[0, 1] = 2.2
    data_test[1, 1] = 1.2
    # class2
    label_test[2:4, 0] = 2
    data_test[2, 0] = 1.7
    data_test[3, 0] = 2
    data_test[2, 1] = 0
    data_test[3, 1] = 0.5
    # class3
    label_test[4:6, 0] = 3
    data_test[4, 0] = 3
    data_test[5, 0] = 0.5
    data_test[4, 1] = 1.
    data_test[5, 1] = 1.5
    return data, label, data_test, label_test


# =========(step1) Determine the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# =========(step2) Generate data from toy example function
n_data = 11;
n_feature = 2;
n_class = 3;
n_test = 6
data_example, label_example, data_example_test, label_example_test = toy_example(n_data, n_feature, n_class, n_test)
# plot data points in a diagram
plt.subplot(2, 1, 1)
data_example1 = data_example.cpu().detach().numpy()
data_example_test1 = data_example_test.cpu().detach().numpy()
plt.plot(data_example1[0:3, 0], data_example1[0:3, 1], '.r')
plt.plot(data_example1[3:7, 0], data_example1[3:7, 1], '.b')
plt.plot(data_example1[7:11, 0], data_example1[7:11, 1], '.g')
plt.plot(data_example_test1[0:2, 0], data_example_test1[0:2, 1], '*r')
plt.plot(data_example_test1[2:4, 0], data_example_test1[2:4, 1], '*b')
plt.plot(data_example_test1[4:6, 0], data_example_test1[4:6, 1], '*g')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("class1(red) class2 (blue) class3(green) test (star points)")
# (satpe1)defining an instant object from the toy example dataset
instance = Kalhor_SeparationIndex(data_example, label_example)
# =========(step3) applying some methods on the object
print('====== methods: "si" and "si_class" ========')
si = instance.si()
print('Separation Index is: ', si.detach().cpu().numpy())
#
si_class = instance.si_class()
print('class Separation Index is: ', si_class.detach().cpu().numpy())
# =================================================
# methods: "high_order_si" and "high_order_si_class"
print('=====methods: "high_order_si" and "high_order_si_class" ========')
order = 1
high_si = instance.high_order_si(order)

print('high Separation Index with order', order, ' is:', high_si.detach().cpu().numpy())

high_si_class = instance.high_order_si_class(order)
print('high Separation Index of classes with order', order, 'are:', high_si_class.detach().cpu().numpy())
# =================================================
# methods: "anti_si" and "anti_si_class"
print('=====methods: "anti_si" and "anti_si_class" ========')
order = 2
anti_si = instance.anti_si(order)
print('anti Separation Index with order', order, ' is:', anti_si.detach().cpu().numpy())
anti_si_class = instance.anti_si_class(order)
print('anti Separation Index of classes with order', order, 'are:', anti_si_class.detach().cpu().numpy())
# =================================================
# methods: "soft_order_si" and "soft_order_si_class"
print('=====methods: "soft_order_si" and "soft_order_si_class" ========')
order = 3
soft_si = instance.soft_order_si(order)
print('soft Sepration Index with order', order, ' is:', soft_si.detach().cpu().numpy())
soft_si_class = instance.soft_order_si_class(order)
print('soft Sepration Index of classes with order', order, 'are:', soft_si_class.detach().cpu().numpy())

# =================================================
# methods: "soft_order_si" and "soft_order_si_class"
print('=====methods: "soft_order_si" and "soft_order_si_class" ========')
print('=====Center SI========')
csi = instance.center_si()
print('center epration Index is: ', csi.detach().cpu().numpy())

csi_class = instance.center_si_class()

print('class center Sepration Index is: ', csi_class.detach().cpu().numpy())

# =================================================
# methods: "cross si" and "cross_si_class", triplet_global_si, triplet_local_si
print('=====methods: "cross si" and "cross_si_class" ========')

print('=====cross SI========')
cross_si = instance.cross_si(data_example_test, label_example_test)
print('Cross Sepration Index is: ', cross_si.detach().cpu().numpy())

cross_si_class = instance.cross_si_class(data_example_test, label_example_test)

print('Cross class Sepration Index is: ', cross_si_class.detach().cpu().numpy())

print('=====Global SI========')
g_si, ancher, postive, negative = instance.triplet_global_si()
print('Global Separation Index is: ', g_si.detach().cpu().numpy())

print('=====Local SI========')
l_si, ancher, postive, negative = instance.triplet_local_si()
print('local Separation Index is: ', l_si.detach().cpu().numpy())
#========================

data_score = instance.data_score_si_anti_si()
print('data scores including maximum si or anti_si of each data point is:',data_score.detach().cpu().numpy())
# cross_data_score = instance.cross_data_score_si_anti_si(data_example_test, label_example_test)


# =================================================
# # method: "data dividing_si"
print('=====method: "data dividing" ========')
[data_difficult, label_difficult, data_easy, label_easy] = instance.data_dividing_si(instance.si_data())
# # X_easy denotes all data points which have equal labels with their nearest neighbors
# # X_difficult denotes all data points which do not have equal labels with their nearest neighbors
#
# # =================================================
# # method: "single_feature_si "
print('=====method: "single_feature_si " ========')
single_feature_si_vec = instance.single_feature_si()
print('Single feature si are: ', single_feature_si_vec.t())
#
# # =================================================
# # method: "forward_feature_ranking_by_si "
print('=====method: "forward_feature_ranking_by_si " ========')

disturbance = torch.randn(n_data, 10, device=device) * 50
data_example_disturbance = torch.concat((data_example, disturbance), 1)
instance_disturbance = Kalhor_SeparationIndex(data_example_disturbance, label_example)
si_ranked_features, ranked_features = instance_disturbance.forward_feature_ranking_si()
plt.subplot(2, 1, 2)
plt.plot(si_ranked_features.cpu().detach().numpy(), 'b')
plt.xlabel("Number of selected features")
plt.ylabel("SI")
plt.title("Forward feature ranking by SI")
si_ranked_features = torch.transpose(si_ranked_features, 0, 1)
print('Ranked features are: ', ranked_features)
print('si for the best chosen Features are: ', si_ranked_features.detach().cpu().numpy())
# # =================================================
# # method: methods: "best_features_forward_si and "ranked_features_best "
print('=====methods: "best_features_forward_si and "ranked_features_best " ========')
data_best_si, ranked_features_best = instance_disturbance.get_best_features_forward_si()
instance_best = Kalhor_SeparationIndex(data_best_si, label_example)
print('Sepration Index for data_best_si is: ', instance_best.si().detach().cpu().numpy())
print('the best feachers are: ', ranked_features_best.t())
# # ==================================================
#
plt.show()

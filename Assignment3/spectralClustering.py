#-*- coding:utf-8 -*-
#!bin/bash
import numpy as np
import datetime
from kmedoid import calc_dis_mat, kmedoids, calc_purity_and_gini_coefficient
import gc

"""
function: 求k近邻
input:    数据矩阵, k值
output:   邻接矩阵(k近邻为1,否则为0)
"""
def knn(data_mat, k):
    row = data_mat.shape[0]
    W = np.zeros((row, row))
    for i in range(row):
        diff_mat = np.tile(data_mat[i], (row, 1)) - data_mat
        variance_mat = diff_mat ** 2
        dis_list = variance_mat.sum(axis=1)**0.5
        sortlist = dis_list.argsort()
        del diff_mat
        del variance_mat
        del dis_list
        print(i, sortlist[0:k+1:1])
        for j in range(1, k+1):
            W[i][sortlist[j]] = 1
            W[sortlist[j]][i] = 1

    del data_mat
    print(W)
    gc.collect()
    return W


"""
function: 计算由邻接矩阵W生成的拉普拉斯矩阵L的特征值和特征向量
input:    邻接矩阵W
output:   最小的k-1个特征值（将特征值为0的特征向量剔除）对应的特征向量构成的矩阵
"""
def calc_L_eig(W, k):
    print(W)
    D = np.diag(W.sum(axis=1))
    print (D)
    # D = np.diag(D)
    # print(D)
    L = D - W
    print(L)
    del D
    del W
    gc.collect()

    eig_vals, eig_vecs = np.linalg.eigh(L)

    print(eig_vals)
    print(eig_vecs)
    sort_index = np.argsort(eig_vals)
    print(sort_index)
    eig_vecs_mat = eig_vecs[sort_index[1:k:1]].T
    print(eig_vecs_mat)
    del eig_vals
    del eig_vecs
    del sort_index
    del L

    return eig_vecs_mat


"""
function: 数据准备
input:    数据文件路径, 文件名
output:   数据标签, 行数, 距离矩阵
"""
def data_pre(base_path, file_name, n, cluster_num):
    data_mat = np.genfromtxt(base_path + file_name, delimiter=",")
    labels = data_mat[:, -1]
    data_mat = data_mat[:, 0: data_mat.shape[1] - 1: 1]
    W = knn(data_mat, n)
    eig_vecs_mat = calc_L_eig(W, cluster_num)
    del W
    dis_mat = np.array(calc_dis_mat(eig_vecs_mat))
    del eig_vecs_mat
    print(dis_mat)
    gc.collect()
    return labels, dis_mat


if __name__ == "__main__":
    base_path = "/home/jun/Desktop/Assignment3/"
    file_names = ["mnist.txt", "german.txt"]
    f1 = open(base_path+"ressc.txt","w")
    cluster_num = 2
    for file_name in file_names:
        if file_name == "german.txt":
            cluster_num = 2
        else:
            cluster_num = 10
        for n in [3, 6, 9]:
            labels, dis_mat = data_pre(base_path, file_name, n, cluster_num)
            for i in range(10):
                start_time = datetime.datetime.now()
                # labels, dis_mat = data_pre(base_path, file_name, n, cluster_num)
                cluster_center, cluster_res, min_cost = kmedoids(cluster_num, dis_mat)
                del dis_mat
                gc.collect()
                purity, gini_coefficient = calc_purity_and_gini_coefficient(cluster_num,
                                                                            cluster_res,
                                                                            labels)
                print(file_name, "'s cluster(" , n, "nn) results is: ")
                print("cluster_center is:", cluster_center, "\npurity is:",
                      purity, "\ngini_coefficient is:", gini_coefficient,
                      "\nmincost is:", min_cost, file)
                end_time = datetime.datetime.now()
                print("round time:", (end_time - start_time))
                del cluster_res
                del cluster_center
                gc.collect()


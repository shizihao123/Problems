#!/usr/bin/env python
import numpy as np
import math


def mds(dist_mat, row, k):                                # 利用MDS实现降维
    dist_i = np.zeros(row)
    for i in range(0, row):
        for j in range(0, row):
            dist_i[i] += pow(dist_mat[i][j], 2)
        dist_i[i] = dist_i[i] / row

    dist_j = np.zeros(row)
    for j in range(0, row):
        for i in range(0, row):
            dist_j[j] += pow(dist_mat[i][j], 2)
        dist_j[j] = dist_j[j] / row

    dist_i_j = 0
    for i in range(0, row):
        for j in range(0, row):
            dist_i_j += pow(dist_mat[i][j], 2)
        dist_i_j = (dist_i_j / pow(row, 2))

    B = np.zeros((row, row))
    for i in range(0, row):
        for j in range(0, row):
            B[i][j] = -0.5*(math.pow(dist_mat[i][j], 2) - dist_i[i] - dist_j[j] + dist_i_j)

    eig_vals, eig_vects = np.linalg.eigh(np.mat(B))
    eig_val_indice = np.argsort(eig_vals)                # 对特征值从小到大排序
    k_eig_val_indice = eig_val_indice[-1:-(k + 1):-1]    # 最大的k个特征值的下标
    k_eig_vect = eig_vects[:, k_eig_val_indice]          # 最大的k个特征值对应的特征向量
    k_eig_vals = eig_vals[k_eig_val_indice]
    for i in range(0, k):
        k_eig_vals[i] = math.sqrt(k_eig_vals[i])
    v = np.diag(k_eig_vals)

    return k_eig_vect * v


def data_pre(base_path, file_name, min_dis_mat_path):                             # 从文本加载数据进行预处理
    # 输入数据准备
    train_path = base_path + file_name + "-train.txt"
    test_path = base_path + file_name + "-test.txt"
    train_mat = np.genfromtxt(train_path, delimiter=',')
    test_mat = np.genfromtxt(test_path, delimiter=',')
    train_labels = train_mat[:, train_mat.shape[1] - 1]
    test_labels = test_mat[:, train_mat.shape[1] - 1]
    min_dis_mat = np.genfromtxt(min_dis_mat_path, delimiter=' ')
    return min_dis_mat, train_labels, test_labels


def _1nn(low_dim_mat, train_labels, test_labels):
    test_line = test_labels.shape[0]
    start_line = train_labels.shape[0]
    predict_labels = np.zeros(test_line)
    low_dim_mat = np.mat(low_dim_mat)
    for i in range(start_line, start_line + test_line):
        dis = 0xfffffff
        for j in range(0, start_line):
            tmp_dis = np.linalg.norm(low_dim_mat[i] - low_dim_mat[j])  # 求两个向量的欧式距离，即二范数
            if tmp_dis < dis:                                          # 记录距离最近的点
                dis = tmp_dis
                predict_labels[i - start_line] = train_labels[j]

    same_num = 0                                                       # 利用1NN预测测试集标签并且计算accuracy
    for i in range(0, predict_labels.shape[0]):
        if predict_labels[i] == test_labels[i]:
            same_num += 1
    return same_num / test_labels.shape[0]


def run(min_dis_mat, train_labels, test_labels, k, data_set_name):     # 执行流程
    low_dim_mat = mds(min_dis_mat, min_dis_mat.shape[0], k)
    accuracy = _1nn(low_dim_mat, train_labels, test_labels)
    print("The accuracy of 1NN prediction of data_set_name", data_set_name, "based on ISOMAP to", k, "dimesions is: ", accuracy)

if __name__ == "__main__":
    base_path = "/home/jun/Desktop/data mining/Assigment2/"
    for file_name in ["sonar", "splice"]:
        min_dis_mat_path = "/home/jun/Desktop/" + file_name + "MinPath.txt"
        min_dis_mat, train_labels, test_labels = data_pre(base_path, file_name, min_dis_mat_path)
        for k in [10, 20, 30]:
            run(min_dis_mat, train_labels, test_labels, k, file_name)










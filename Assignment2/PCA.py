#!/usr/bin/env python
import numpy as np


def zero_mean(in_matrix):                                          # 按列求均值，即求各个特征的均值
    mean_val = np.mean(in_matrix, axis = 0)
    out_matrix = in_matrix - mean_val
    return out_matrix, mean_val


def cov(data_mat):                                              # 求协方差矩阵
    res = np.zeros((data_mat.shape[1], data_mat.shape[1]))
    for i in range(0, data_mat.shape[1]):
        for j in range(0, data_mat.shape[1]):
            res[i][j] = np.dot(data_mat[:, i], data_mat[:, j]) / (data_mat.shape[0] - 1)
    return res


def pca(data_matrix, k):                                           # PCA降维
    new_data, mean_data = zero_mean(data_matrix)
    cov_mat = cov(new_data)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    eig_val_indice = np.argsort(eig_vals)                            # 对特征值从小到大排序
    k_eig_val_indice = eig_val_indice[-1:-(k + 1):-1]                # 最大的k个特征值的下标
    k_eig_vects = eig_vects[:, k_eig_val_indice]                      # 最大的k个特征值对应的特征向量
    return k_eig_vects                                              # 返回投影矩阵


def _1nn(pj_train, pj_test, train_labels, test_labels):            # 利用1NN预测测试集标签并且计算accuracy
    predict_labels = np.zeros(pj_test.shape[0])
    for i in range(0, pj_test.shape[0]):
        dis = 0xfffffff
        for j in range(0, pj_train.shape[0]):
            tmp_dis = np.linalg.norm(pj_test[i] - pj_train[j])     # 求两个向量的欧式距离，即二范数
            if tmp_dis < dis:                                      # 记录距离最近的点
                dis = tmp_dis
                predict_labels[i] = train_labels[j]

    same_num = 0
    for i in range(0, predict_labels.shape[0]):
            if predict_labels[i] == test_labels[i]:
                same_num += 1
    return same_num / test_labels.shape[0]


def data_pre(base_path, file_name):                                       # 从文本加载数据进行预处理
    # 输入数据准备
    path_train = base_path + file_name + "-train.txt"
    path_test = base_path + file_name + "-test.txt"
    data_train = np.genfromtxt(path_train, delimiter=',')
    data_test = np.genfromtxt(path_test, delimiter=',')
    train_labels = data_train[:, data_train.shape[1] - 1]
    test_labels = data_test[:, data_train.shape[1] - 1]
    data_train = np.delete(data_train, data_train.shape[1] - 1, axis=1)  # 删除标签列
    data_test = np.delete(data_test, data_test.shape[1] - 1, axis=1)     # 删除标签列
    return data_train, data_test, train_labels, test_labels


def run(data_train, data_test, train_labels, test_labels, k, data_set):  # 执行流程
    pj_matrix = pca(data_train, k)  # 获得投影矩阵
    new_data1, mean_val = zero_mean(data_train)
    new_data2, mean_val = zero_mean(data_test)
    pj_train = new_data1 * pj_matrix
    pj_test = new_data2 * pj_matrix
    accuracy = _1nn(pj_train, pj_test, train_labels, test_labels)
    print("The accuracy of 1NN prediction of data_set", data_set, "based on pca to", k, "dimesions is: ", accuracy)


if __name__ == "__main__":
    base_path = "/home/jun/Desktop/data mining/Assigment2/"
    for file_name in ["sonar", "splice"]:
        data_train, data_test, train_labels, test_labels = data_pre(base_path, file_name)
        for k in [10, 20, 30]:
            run(data_train, data_test, train_labels, test_labels, k, file_name)








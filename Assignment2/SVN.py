#!/usr/bin/env python
import numpy as np


def svn(data_matrix, k):                                                     # svn分解实现降维
    [u, s, v] = np.linalg.svd(data_matrix)
    p = v[range(0, k), :]
    return p.T


def data_pre(base_path, file_name):                                          # 从文本加载数据进行预处理
    # 输入数据准备
    train_path = base_path + file_name + "-train.txt"
    test_path = base_path + file_name + "-test.txt"
    train_data = np.genfromtxt(train_path, delimiter=',')
    test_data = np.genfromtxt(test_path, delimiter=',')
    train_labels = train_data[:, train_data.shape[1] - 1]
    test_labels = test_data[:, train_data.shape[1] - 1]
    train_data = np.delete(train_data, train_data.shape[1] - 1, axis=1)      # 删除标签列
    test_data = np.delete(test_data, test_data.shape[1] - 1, axis=1)
    return train_data, test_data, train_labels, test_labels


def _1nn(pj_train_mat, pj_test_mat, train_labels, test_labels):
    predict_labels = np.zeros(pj_test_mat.shape[0])
    for i in range(0, pj_test_mat.shape[0]):
        dis = 0xfffffff
        for j in range(0, pj_train_mat.shape[0]):
            tmp_dis = np.linalg.norm(pj_test_mat[i] - pj_train_mat[j])      # 求两个向量的欧式距离，即二范数
            if tmp_dis < dis:                                               # 记录距离最近的点
                dis = tmp_dis
                predict_labels[i] = train_labels[j]

    same_num = 0                                                            # 利用1NN预测测试集标签并且计算accuracy
    for i in range(0, predict_labels.shape[0]):
            if predict_labels[i] == test_labels[i]:
                same_num += 1
    return same_num / test_labels.shape[0]


def run(train_mat, test_mat, train_labels, test_labels, k, data_set_name):  # 执行流程
    pj_matrix = svn(train_mat, k)  # 获得投影矩阵
    pj_train_mat = np.dot(train_mat, pj_matrix)
    pj_test_mat = np.dot(test_mat, pj_matrix)
    accuracy = _1nn(pj_train_mat, pj_test_mat, train_labels, test_labels)
    print("The accuracy of 1NN prediction of data_set_name", data_set_name, "based on svn to", k, "dimesions is: ", accuracy)

if __name__ == "__main__":
    base_path = "/home/jun/Desktop/data mining/Assigment2/"
    for file_name in ["sonar", "splice"]:
        train_mat, test_mat, train_labels, test_labels = data_pre(base_path, file_name)
        for k in [10, 20, 30]:
            run(train_mat, test_mat, train_labels, test_labels, k, file_name)
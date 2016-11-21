# !bin/bash
#  -*- coding:utf-8 -*-
import numpy as np
import math
PI = 3.1415926
# Todo: 参数平滑,待解决
# Todo: 连续值转高斯分布


def data_pre(base_path, file_name):
    data_mat = np.genfromtxt(base_path + file_name, delimiter=",")
    train_y = data_mat[1:data_mat.shape[0]:1, -1]
    types_row = data_mat[0]
    train_x = data_mat[1: data_mat.shape[0] : 1, 0 : data_mat.shape[1]-1 : 1]
    return train_y, train_x, types_row


def prob_pre(train_x, train_y, types_row):
    print(train_x.shape, train_y.shape)
    N = train_x.shape[0]
    M = train_x.shape[1]
    types_num = {}
    types_index = {}
    for i in range(N):
        if types_num.__contains__(train_y[i]):
            types_num[train_y[i]] += 1.0
            types_index[train_y[i]].append(i)
        else:
            types_num[train_y[i]] = 1.0
            types_index[train_y[i]] = []
            types_index[train_y[i]].append(i)

    p_yk = {}
    for key in types_num.keys():
        p_yk[key] = float(types_num[key] / N)

    p_xi_yk = {}
    con_val_ave = {}
    con_val_variace = {}
    for key in types_num.keys():
        p_xi_yk[key] = {}
        con_val_ave[key] = {}
        con_val_variace[key] = {}
        print (key)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    for j in range(M):
        if types_row[j] != 1:
            for i in range(N):
                for key in types_num.keys():
                    con_val_ave[key][j] = float(train_x[types_index[key], j].sum(axis=0)) / N
                    # print (key, j, types_index[key])
                    # print "hello" , pow(np.array(train_x[types_index[key], j]).T - con_val_ave[key][j], 2).shape
                    con_val_variace[key][j] = pow(np.array(train_x[types_index[key], j]) - con_val_ave[key][j], 2).sum() / N


    for i in range(N):
        for j in range(M):
            if types_row[j] != 1:
                if p_xi_yk[train_y[i]].__contains__(train_x[i][j]):
                    p_xi_yk[train_y[i]][train_x[i][j]] += 1.0
                else:
                    p_xi_yk[train_y[i]][train_x[i][j]] = 2.0
            # else:
            #     if p_xi_yk[train_y[i]].__contains__(train_x[i][j]):
            #         p_xi_yk[train_y[i]][train_x[i][j]] =\
            #             1.0 / np.sprt(2.0*math.pi * con_val_variace[j])\
            #         *math.exp(-1 * pow((train_x[i][j] - con_val_ave[j]),2) / (2.0 * con_val_variace[j]))


    for key in types_num.keys():
        for key2 in p_xi_yk[key].keys():
            p_xi_yk[key][key2] = float(p_xi_yk[key][key2] / types_num[key])

    return p_xi_yk, p_yk, con_val_ave, con_val_variace


def test_validate(test_x, test_y, p_xi_yk, p_yk, train_num, types_row, con_val_ave, con_val_variace):
    predict_y = []
    predict_correct = 0
    i = 0
    print ("enter test:")
    print (test_x.shape, test_y.shape)
    print (p_xi_yk)
    print (p_yk)
    for instance in test_x:
        p_test = 0.0
        predict_label = None
        for key in p_yk.keys():
            p_tmp = 1.0 * p_yk[key]
            for j in range(instance.shape[0]):
                if types_row[j] != 1:
                    p_xi_yk[key][instance[j]] = \
                        1.0 / math.sqrt(2.0 * math.pi * con_val_variace[key][j]) \
                        * math.exp(-1 * pow((instance[j] - con_val_ave[key][j]), 2) / (2.0 * con_val_variace[key][j]))
                if not p_xi_yk[key].__contains__(instance[j]):
                    p_xi_yk[key][instance[j]] = 1.0 / (train_num + 1)
                p_tmp = p_tmp * p_xi_yk[key][instance[j]]
            if p_tmp > p_test:
                p_tmp = p_test
                predict_label = key
        predict_y.append(predict_label)
        if predict_label == test_y[i]:
            predict_correct += 1
        i += 1
    print predict_y
    print test_y
    accuray = float(predict_correct) / float(test_x.shape[0])
    print(accuray)
    return accuray


def ten_fold_cross_validation_split(train_x_origin, train_y_origin, test_split = 9):
    N = train_x_origin.shape[0]
    print N
    ave = N / 10
    test_x = train_x_origin[ave * (test_split - 1): ave * test_split - 1: 1, :]
    test_y = train_y_origin[ave * (test_split - 1): ave * test_split - 1: 1]

    print test_x.shape, test_y.shape
    print np.array(train_x_origin[0:ave * (test_split - 1):1, :]).shape
    print np.array(train_x_origin[ave * test_split - 1:N:1, :]).shape
    train_x = np.concatenate([np.array(train_x_origin[0:ave * (test_split - 1):1, :]), np.array(train_x_origin[ave * test_split - 1:N:1, :])])
    print train_x.shape
    print  train_x_origin[ave * test_split - 1:N:1].shape
    train_y = np.concatenate([(list)(train_y_origin[0:ave * (test_split - 1):1]), list(train_y_origin[ave * test_split - 1:N:1])])

    return train_x, train_y, test_x, test_y, N/10*9


if __name__ == '__main__':
    file_names = ["german-assignment5.txt", "breast-cancer-assignment5.txt"]
    train_y, train_x, types_row = data_pre("./", "german-assignment5.txt")
    train_x, train_y, test_x, test_y, train_num = ten_fold_cross_validation_split(train_x, train_y, 4)
    p_xi_yk, p_yk, con_val_ave, con_val_variace = prob_pre(train_x, train_y, types_row)
    accuracy = {}
    accuray = test_validate(test_x, test_y, p_xi_yk, p_yk, train_num, types_row, con_val_ave, con_val_variace)
    # a = np.array([2,3,4]).T
    # b = a + 3
    # print (b)
    #
    # print pow(b,2).sum()
    # a = np.array([2,3,4,5,6,7])
    # b = [1,2,3]
    # print a[[1,3]]





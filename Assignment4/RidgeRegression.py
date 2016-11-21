#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random

import sklearn.linear_model as Lr

from Assignment4 import LogisticRegression as LR
from Assignment4.LogisticRegression import plot


def sklearn_try(train_x, train_y,test_x, test_y, weights = None, lamda = 1):
    lr = Lr.SGDClassifier(loss='log', penalty='none', alpha= 0.1, random_state=5)
    print train_x.shape, train_y.shape
    lr.fit(train_x, train_y)
    print 'train score:', lr.score(train_x, train_y)
    print 'test score:', lr.score(test_x, test_y)


def calcAccuracy(weights, data_mat_x, data_mat_y):
    numSamples, numFeatures = data_mat_x.shape
    matchCount = 0
    # print "numSamples", numSamples
    for i in range(numSamples):
        predict = np.dot(data_mat_x[i, :], weights) > 0
        # print predict
        # print "predict", sigmoid(np.dot(data_mat_x[i, :], weights))
        # print "data_mat_y", i, data_mat_y[i]
        if predict == bool(data_mat_y[i] > 0):
            # print "hello"
            matchCount += 1
    # print matchCount, numSamples
    accuracy = float(matchCount) / float(numSamples)
    # print "accuracy", accuracy
    return accuracy


def calc_obj_func(weights, train_x, train_y, lamda = 1):
    objective = 0
    # print train_x.shape
    numSamples, numFeatures = train_x.shape
    # print "start pbjective", objective
    for i in range(numSamples):
        # print weights
        # print (lamda * np.linalg.norm(weights))
        # print (sigmoid(-1 * train_y[i] * np.dot(train_x[i, :], weights)))
        # objective += sigmoid(-1 * train_y[i] * np.dot(train_x[i, :], weights))
        objective += pow(train_y[i] - np.dot(train_x[i, :], weights), 2)
        # print "objecttive", objective
    objective = float(objective / numSamples) + lamda * np.linalg.norm(weights)
    return objective


def train_rr(weights, train_x, train_y, test_x, test_y, opts):
    numSamples, numFeatures = train_x.shape
    alpha = opts['alpha']                      # alpha为更新步长
    lamda = opts['lamda']                      # lamda为正则化项的系数

    indices = range(numSamples)                # 随机打乱样本集index以进行随机梯度下降
    random.shuffle(indices)
    for i in indices:
        predict = np.dot(train_x[i, :], weights)  # 计算预测值
        weights -= alpha * (2*(predict - train_y[i])*np.array([train_x[i, :]]).T + 2*lamda*weights)  # LR正则化更新权重

    objective = calc_obj_func(weights, train_x, train_y, lamda)
    accuracy_train = calcAccuracy(weights, train_x, train_y)
    accuracy_test = calcAccuracy(weights, test_x, test_y)
    return weights, accuracy_train, accuracy_test, objective


def draw_log_regres(maxIter, train_x, train_y, test_x, test_y, opts):
    numSamples, numFeatures = train_x.shape
    weights = np.ones((numFeatures, 1))/ numFeatures
    # print weights
    # plt.axis([0, maxIter + 1, 0, 1])
    # plt.ion()
    reduce_cycle = opts["alpha_cycle"]
    reduce_times = -1
    lower_bound = opts["lower_bound"]
    error_rate_train = []
    error_rate_test = []
    obj_func = []
    file_name = opts["file_name"]
    save_path = opts["save_path"]
    for i in range(maxIter):
        if opts["alpha"] <= lower_bound:
            opts["alpha"] = lower_bound
        elif i / reduce_cycle > reduce_times:
            opts["alpha"] /= 2.0
            reduce_times += 1

        # sklearn_try(train_x,train_y,test_x, test_y,weights)
        weights, accuracy_train, accuracy_test, objective = train_rr(weights, train_x, train_y, test_x, test_y, opts)
        # print accuracy_train," ", accuracy_test, " ", objective
        error_rate_train.append(1.0 - accuracy_train)
        error_rate_test.append(1.0 - accuracy_test)
        obj_func.append(objective)

        # plt.scatter(i, float(accuracy_train), c='r', marker='o')
        # plt.scatter(i, float(accuracy_test), c='b', marker='o')
        # plt.scatter(i, float(objective), c='g', marker='o')
        # plt.pause(0.05)

    plot(file_name + "_RR", range(maxIter), error_rate_train, error_rate_test, obj_func, save_path, file_name)


if __name__ == "__main__":
    base_path = "/home/jun/Desktop/data mining/Assignment4/"
    file_names = ["dataset1-a9a-training.txt", "covtype-training.txt"]
    file_name = "dataset1-a9a-"
    train_y, train_x = LR.data_pre(base_path, file_name + "training.txt")
    test_y, test_x = LR.data_pre(base_path, file_name + "testing.txt")
    opts = {}
    opts["alpha"] = 0.00001
    opts["lamda"] = 0.00001
    opts["alpha_cycle"] = 10
    opts["lower_bound"] = 0.000001
    opts["save_path"] = "/home/jun/Desktop/data mining/Assignment4/"
    opts["file_name"] = file_name.strip("-")
    # print train_x.shape, train_y.shape
    weights = draw_log_regres(100, train_x, train_y, test_x, test_y, opts)
    # a = np.array([[1],[2],[3]])
    # print np.linalg.norm(a)
    # print (np.sum(a, axis=0), a.shape)

    # b =np.array([[1,3,4],[1,2,3]])
    # print ( np.dot(a,  b))
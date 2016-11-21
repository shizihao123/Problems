# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pylab as pl, matplotlib.pyplot as plt
import time
import random
import sklearn.linear_model as Lr


def sklearn_try(train_x, train_y,test_x, test_y, weights = None, lamda = 1):
    lr = Lr.SGDClassifier(loss='log', penalty='none', alpha= 0.1, random_state=5)
    print train_x.shape, train_y.shape
    lr.fit(train_x, train_y)
    print 'train score:', lr.score(train_x, train_y)
    print 'test score:', lr.score(test_x, test_y)


def calc_accuracy(weights, data_mat_x, data_mat_y):
    numSamples, numFeatures = data_mat_x.shape
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(np.dot(data_mat_x[i, :], weights)) > 0.5
        if predict == bool(data_mat_y[i] > 0):
            matchCount += 1
    accuracy = float(matchCount) / float(numSamples)
    return accuracy


# def save_data(title, iterNums, error_rate_train, error_rate_test, obj_func, save_path, file_name):
#     f = open(save_path + file_name, "w")
#     f.write("title is:", title, "\n")
#     f.write("iterNums is:", iterNums, "\n")
#     f.write("error_rate_train is:", error_rate_train, "\n")
#     f.write("error_rate_test is:", error_rate_test, "\n")
#     f.write("obj_func:", obj_func, "\n")


def plot(title, iterNums, errorRate_train, errorRate_test, objFunc, save_path, file_name):
    # save_data(title, iterNums, errorRate_train, errorRate_test, objFunc, save_path, file_name)
    print "title", title
    print "iterNums", iterNums
    print "errorRate_train", errorRate_train
    print "errorRate_test", errorRate_test
    print "objFunc", objFunc

    pl.plot(iterNums, errorRate_train, 'r-', label='errorRate_train')
    pl.plot(iterNums, errorRate_test, 'b-', label='errorRate_test')
    pl.plot(iterNums, objFunc, 'y-', label='objFunc')
    pl.title(title)
    pl.xlabel('iterNum')
    pl.ylabel('errorRate/objFunc')
    # pl.xlim(0.0,10.0)
    # pl.ylim(2.0,10.0)
    pl.legend()
    pl.show()


def data_pre(base_path, file_name):
    data_mat = np.genfromtxt(base_path + file_name, delimiter=",")
    train_y = data_mat[:, -1]
    extra_x = np.ones((data_mat.shape[0], 1))
    train_x = np.c_[data_mat[:, 0: data_mat.shape[1] - 1: 1], extra_x]
    return train_y, train_x


def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))


def calc_obj_func(weights, train_x, train_y, lamda = 1):
    objective = 0
    # print train_x.shape
    sample_nums, feature_nums = train_x.shape
    # print "start pbjective", objective
    for i in range(sample_nums):
        # print weights
        # print (lamda * np.linalg.norm(weights))
        # print (sigmoid(-1 * train_y[i] * np.dot(train_x[i, :], weights)))
        # objective += sigmoid(-1 * train_y[i] * np.dot(train_x[i, :], weights))
        objective += sigmoid(-1 * train_y[i] * np.dot(train_x[i, :], weights))
        # print "objecttive", objective
    objective = float(objective / sample_nums)  + lamda * np.linalg.norm(weights)
    return objective

def train_lr(weights, train_x, train_y, test_x, test_y, opts):
    num_samples, num_features = train_x.shape
    alpha = opts['alpha']                          # alpha为更新步长
    lamda = opts['lamda']                          # lamda为正则化项的系数

    indices = range(num_samples)
    random.shuffle(indices)                        # 随机打乱样本集index以进行随机梯度下降
    for i in indices:
        X = -1.0 * train_y[i] * np.dot(train_x[i, :], weights)
        incorrect = float(sigmoid(X))              # 计算不正确的概率
        weights -= alpha * (np.array([train_x[i, :]]).T * -1 *train_y[i] * incorrect + 2*lamda*weights) #梯度下降

    objective = calc_obj_func(weights, train_x, train_y, lamda)  # 计算目标函数
    accuracy_train = calc_accuracy(weights, train_x, train_y)     # 计算训练集精确度
    accuracy_test = calc_accuracy(weights, test_x, test_y)        # 计算测试集精确度
    return weights, accuracy_train, accuracy_test, objective


def draw_lr(max_iter, train_x, train_y, test_x, test_y, opts):
    num_samples, num_features = train_x.shape
    weights = np.ones((num_features, 1)) / num_features
    print weights
    # plt.axis([0, max_iter + 1, 0, 1])
    # plt.ion()
    reduce_cycle = opts["alpha_cycle"]
    reduce_times = -1
    lower_bound = opts["lower_bound"]
    error_rate_train = []
    error_rate_test = []
    obj_func = []
    file_name = opts["file_name"]
    save_path = opts["save_path"]

    for i in range(max_iter):
        if opts["alpha"] <= lower_bound:
            opts["alpha"] = lower_bound
        elif i / reduce_cycle > reduce_times:
            opts["alpha"] /= 2.0
            reduce_times += 1

        sklearn_try(train_x,train_y,test_x, test_y,weights)
        weights, accuracy_train, accuracy_test, objective = train_lr(weights, train_x, train_y, test_x, test_y, opts)
        print accuracy_train," ", accuracy_test, " ", objective
        error_rate_train.append(1.0 - accuracy_train)
        error_rate_test.append(1.0 - accuracy_test)
        obj_func .append(objective)
        # plt.scatter(i, float(accuracy_train), c='r', marker='o')
        # plt.scatter(i, float(accuracy_test), c='b', marker='o')
        # plt.scatter(i, float(objective), c='g', marker='o')
        # plt.pause(0.1)

    plot(file_name + "_LR", range(max_iter), error_rate_train, error_rate_test, obj_func, save_path, file_name)

    # 绘图
    # print("------------------------------------dataset1------------------------------------")
    # print("1、logistic regression+++++++++++++++")
    # print("  iterNums: ", LR_iterNums)
    # print("  errorRate_train: ", LR_errorRate_train)
    # print("  errorRate_test: ", LR_errorRate_test)
    # print("  objFunc: ", LR_objFunc)
    #
    # print("2、ridge regression+++++++++++++++")
    # print("  iterNums: ", RR_iterNums)
    # print("  errorRate_train: ", RR_errorRate_train)
    # print("  errorRate_test: ", RR_errorRate_test)
    # print("  objFunc: ", RR_objFunc)

if __name__ == "__main__":
    base_path = "/home/jun/Desktop/data mining/Assignment4/"
    file_names = ["dataset1-a9a-training.txt", "covtype-training.txt"]
    file_name = "covtype-"
    train_y, train_x = data_pre(base_path, file_name + "training.txt")
    test_y, test_x = data_pre(base_path, file_name + "testing.txt")
    opts = {}
    opts["alpha"] = 0.0001
    opts["lamda"] = 0.0001
    opts["alpha_cycle"] = 10
    opts["lower_bound"] = 0.00001
    opts["save_path"] = "/home/jun/Desktop/data mining/Assignment4/"
    opts["file_name"] = file_name.strip("-")
    print train_x.shape, train_y.shape
    weights = draw_lr(100, train_x, train_y, test_x, test_y, opts)

    # a = np.array([[1],[2],[3]])
    # print np.linalg.norm(a)
    # print (np.sum(a, axis=0), a.shape)

    # b =np.array([[1,3,4],[1,2,3]])
    # print ( np.dot(a,  b))
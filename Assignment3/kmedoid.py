#-*- coding:utf-8 -*-
#!bin/bash
import numpy as np
import random
import datetime

"""
function: 数据准备
input:    数据文件路径, 文件名
output:   数据标签, 行数, 距离矩阵"""
def data_pre(base_path, file_name):
    data_mat = np.genfromtxt(base_path + file_name, delimiter=",")
    labels = data_mat[:, -1]
    data_mat = data_mat[:, 0: data_mat.shape[1] - 1: 1]
    dis_mat = calc_dis_mat(data_mat)
    return labels, dis_mat


"""
function: 计算距离矩阵
input:    样本数据矩阵
output:   距离矩阵
"""


def calc_dis_mat(data_mat):
    row = data_mat.shape[0]
    dis_mat = []
    for i in range(row):
        diff_mat = np.tile(data_mat[i], (row, 1)) - data_mat
        abs_diff_mat = np.abs(diff_mat)
        dis_vec = abs_diff_mat.sum(axis=1)
        dis_mat.append(dis_vec)
        print(np.array(dis_mat).shape)
    return dis_mat


"""
function: 分配所有点到最近的聚类中心
input:    聚类中心, 聚类个数, 距离矩阵
output:   代价和聚类结果
"""


def medoids_arange(cluster_center,
                   cluster_num,
                   dis_mat):
    instance_num = np.array(dis_mat).shape[0]

    cluster_cost = np.zeros(cluster_num)
    cost = 0
    cluster_res = {}
    for k in range(cluster_num):
        cluster_res[k] = []

    for i in range(0, instance_num):
        min_dis = 0xfffffff
        belong_cluster = 0
        for k in range(0, cluster_num):
            if dis_mat[i][cluster_center[k]] < min_dis:
                belong_cluster = k
                min_dis = dis_mat[i][cluster_center[k]]
        cost += min_dis  # 代价cost为计算所有点到各自聚类中心的距离和, 也是迭代优化的目标函数
        cluster_res[belong_cluster].append(i)  # 聚类结果
        cluster_cost[belong_cluster] += min_dis

    return cluster_cost, cost, cluster_res


"""
function: kmedoids主方法
input:    聚类个数, 距离矩阵
output:   最终聚类中心, 聚类结果, 代价
"""
def kmedoids(cluster_num,
             dis_mat):
    instance_num = np.array(dis_mat).shape[0]
    cluster_center = []
    old_center = []

    for i in range(0, cluster_num):  # 随机初始化集cluster_num个聚类中心
        cluster_center.append(
            random.randint(int(instance_num / cluster_num) * i,
                           int(instance_num / cluster_num) * (i + 1) )
        )
        old_center.append(cluster_center[i])

    cluster_cost , prev_cost, cluster_res = medoids_arange(
        cluster_center,
        cluster_num,
        dis_mat
    )

    while True:
        for k in range(0, cluster_num):  # 从k个聚类中分别尝试用该类中不同点替换中心点, 看是否能够减少代价
            for i in range(0, cluster_res[k].__len__()):
                if i == cluster_center[k]:
                    continue
                temp = cluster_center[k]
                cluster_center[k] = i
                # now_cost, tmp_res = medoids_arange(  # 重新分配所有点, 计算新中心的代价
                #     cluster_center,
                #     cluster_num,
                #     dis_mat)

                tmp_cost = 0 
                for element in cluster_res[k]:
                    tmp_cost += dis_mat[element][cluster_center[k]]
                if tmp_cost < cluster_cost[k] : # 代价降低则接受
                    cluster_cost[k] = tmp_cost
                else:
                    cluster_center[k] = temp  # 丢弃新中心, 找回原中心

        if old_center == cluster_center:  # 新一轮迭代, 聚类中心不变则结束
            print(cluster_center)
            print(old_center)
            break
        else:
            for i in range(0, cluster_num):  # 更新旧聚类中心
                old_center[i] = cluster_center[i]
            print(old_center)
            cluster_cost, prev_cost, cluster_res = medoids_arange(cluster_center,cluster_num,dis_mat)


    return cluster_center, cluster_res, prev_cost  # 返回聚类中心, 聚类结果, 代价


"""
function: 计算purity 和 gini_coefficient
input: 聚类个数, 聚类结果, 数据真正标签类别
ouput: 纯度和基尼指数
"""
def calc_purity_and_gini_coefficient(cluster_num, cluster_res, labels):
    pij_num = 0
    Gj = {}
    for i in range(cluster_num):
        # print(cluster_res[i])
        max_type = 0
        label_dict = {}
        variance_sum = 0
        for label in labels:
            label_dict[label] = 0
        for index in cluster_res[i]:
            label_dict[labels[index]] += 1
            if (label_dict[labels[index]]) > max_type:
                max_type = label_dict[labels[index]]
        for key in label_dict.keys():
            variance_sum += pow(label_dict[key] / cluster_res[i].__len__(), 2)
        Gj[cluster_res[i].__len__()] = 1 - variance_sum
        pij_num += max_type

    purity = 0
    for i in range(cluster_num):
        purity = pij_num / labels.__len__()
    # print(purity, min_cost)

    denominator = 0
    numerator = 0
    for key in Gj.keys():
        numerator += key * Gj[key]
        denominator += key
    gini_coefficient = float(numerator / denominator)

    # print(gini_coefficient)
    return purity, gini_coefficient


if __name__ == "__main__":
    base_path = "/home/jun/Desktop/Assignment3/"
    file_names = ["mnist.txt"]
    # f1 = open(base_path + "reskm.txt", "w")
    cluster_num = 2
    for file_name in file_names:
        if file_name == "german.txt":
            cluster_num = 2
        else:
            cluster_num = 10
        labels, dis_mat = data_pre(base_path, file_name)
        for i in range(10):
            start_time = datetime.datetime.now()
            # l
            # abels, dis_mat = data_pre(base_path, file_name)
            cluster_center, cluster_res, min_cost = kmedoids(cluster_num, dis_mat)
            purity, gini_coefficient = calc_purity_and_gini_coefficient(cluster_num,
                                                                        cluster_res,
                                                                        labels)
            print(file_name, "'s cluster", " results is: ")
            print("cluster_center is:", cluster_center, "\npurity is:",
                  purity, "\ngini_coefficient is:",
                  gini_coefficient, "\nmincost is:", min_cost)
            end_time = datetime.datetime.now()
            print("round time: ", (end_time - start_time))



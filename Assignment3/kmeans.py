
# #!bin/bash
# import  numpy as np
# import random
# import  string
# import  datetime
#
# # def dis(vec1, vec2):
# #     len = vec1.__len__()
# #     res = 0
# #     for i in range(0, len):
# #         res += abs(vec1[i] - vec2[i])
# #     return res
#
# def distance(data_mat):
#     row = data_mat.shape[0]
#     dis_mat = []
#     # f = open("/home/jun/Desktop/dis_mat.txt", "ｒ")
#     for i in range(row):
#         diff_mat = np.tile(data_mat[i], (row, 1)) - data_mat
#         abs_diff_mat = np.abs(diff_mat)
#         dis_vec = abs_diff_mat.sum(axis=1)
#         dis_mat.append(dis_vec)
#         print(np.array(dis_mat).shape)
#
#         # for index in range(row - 1):
#         #         print(dis_vec[index], end=" ", file=f)
#         # print(dis_vec[row-1],file=f)
#
#     # f.close()
#     return  dis_mat
#
#
# def medoids_arange(cluster_center,instance_num,cluster_num, dis_mat):
#     cost = 0
#     cluster_res = {}
#     for k in range(cluster_num):
#         cluster_res[k] = []
#     for i in range(0, instance_num):
#         min_dis = 0xfffffff
#         belong_cluster = 0
#         for k in range(0, cluster_num):
#             if (dis_mat[i][cluster_center[k]] < min_dis):
#                 belong_cluster = k
#                 min_dis = dis_mat[i][cluster_center[k]]
#         cost += min_dis
#         cluster_res[belong_cluster].append(i)
#     return cost, cluster_res
#
#
# def kmedoids(data_mat, cluster_num, dis_mat):
#     instance_num = data_mat.shape[0]
#     cluster_center = []
#     old_center = []
#
#     for i in range(0, cluster_num):
#         cluster_center.append(
#             random.randint(int(instance_num / cluster_num) * i, int(instance_num / cluster_num) * (i + 1)))
#         old_center.append(cluster_center[i])
#
#     print(old_center)
#     prev_cost , cluster_res = medoids_arange(cluster_center,instance_num,cluster_num, dis_mat)
#     print(prev_cost, cluster_res)
#
#     while True:
#         for k in range(0, cluster_num):
#             for i in range(0, cluster_res[k].__len__()):
#                 temp = cluster_center[k]
#                 cluster_center[k] = i
#                 now_cost, tmp_res = medoids_arange(cluster_center, instance_num, cluster_num, dis_mat)
#                 print(now_cost, tmp_res)
#                 if(now_cost < prev_cost):
#                     prev_cost = now_cost
#                     cluster_res = tmp_res
#                 else:
#                     cluster_center[k] = temp
#
#         if(old_center == cluster_center):
#             print(cluster_center)
#             print(old_center)
#             break
#         else:
#              for i in range(0, cluster_num):
#                 old_center[i] = cluster_center[i]
#              print(old_center)
#     return  old_center, cluster_res ,prev_cost
#
#
# if __name__ == "__main__":
#     start_time = datetime.datetime.now()
#     base_path = "/home/jun/Desktop/Assignment3/"
#     data_mat = np.genfromtxt(base_path + "mnist.txt", delimiter=",")
#     labels = data_mat[:,-1]
#     data_mat = data_mat[:, 0 : data_mat.shape[1] - 1 : 1]
#     row = data_mat.shape[0]
#     dis_mat = distance(data_mat)
#
#     end_time = datetime.datetime.now()
#     print(dis_mat)
#     print("calc dis_mat time", (end_time - start_time)._seconds)
#
#     cluster_num = 10
#     cluster_center, cluster_res, min_cost = kmedoids(data_mat, cluster_num, dis_mat)
#     p_ij = []
#     for i in range(cluster_num):
#         max_type = 0
#         labelDict = {}
#         for label in labels:
#             labelDict[label] = 0
#         for index in cluster_res[i]:
#             labelDict[labels[index]] += 1
#             if (labelDict[labels[index]]) > max_type:
#                 max_type = labelDict[labels[index]]
#         p_ij.append(max_type / cluster_res[i].__len__())
#
#     purity = 0
#     for i in range(cluster_num):
#         purity += cluster_res[i].__len__() / data_mat.shape[0] *p_ij[i]
#
#     print(purity, min_cost)
#     end_time = datetime.datetime.now()
#     print("finish_time", end_time - start_time)
#
#

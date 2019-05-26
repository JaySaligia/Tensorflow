import math
import numpy as np
from sklearn import metrics
import random

class Node(object):#节点信息
    def __init__(self, point_num):
        self.cluster_v = np.ndarray#每个类的平均向量
        self.point_num = point_num#每个类包含的点

    def terrianinfo(self):#计算每个类的边界信息,采用均值来作为类中心点
        length = len(self.point_num)
        sum_point = a[self.point_num[0]]
        for i in range(1, length):
            sum_point += a[self.point_num[i]]
        self.cluster_v = sum_point / length

def preprocession():#预处理
    #预先计算了每个维度数据的方差与均方差，作为判断数据波动和其在所有维度权重中的依据，根据这个对某些维度的数据进行了加权
    a_label = np.loadtxt('breast.txt')
    label = a_label[:, 9].astype(np.int32)
    a = np.delete(a_label, -1, axis=1)
    a[:, 4] = a[:, 4] * 0.9
    a[:, 8] = a[:, 8] * 0.5
    return a, label

def caldist(vec1, vec2):#计算两点欧式距离
    return np.linalg.norm(vec1 - vec2)

def Hierarchical_Clusteringluster(now):#层次聚类
    #基本思路：每一次聚类，对于没有分类的点找其近邻点，然后（1：若其近邻点属于某一类，将其也归入该类；2：若其近邻点不属于某一类，则其与其近邻点自成一类），每次聚类完，更新每类的中心点，作为新的用于聚类时的点；3：与其最近邻点过远，自己作为一个类）
    length = len(now)
    print("having " + str(length) + " clusters")
    if length == 1:
        return np.zeros(699, dtype = np.int)
    if length == 2:
       ret = np.zeros(699, dtype = np.int)
       for i in range(length):
            for j in now[i].point_num:
                ret[j] = (i+1) * 2
            print(now[i].point_num)
       return ret
    next = []
    groups = [-1 for i in range(length)]#对未分类的类设为-1
    group_count = 0
    dist_count = 0
    dist_total = 0
    for i in range(length):
        dist_min = 100000
        match_num = 0
        if groups[i] == -1:
            for j in range(length):
                if not i == j:
                    dist = caldist(now[i].cluster_v, now[j].cluster_v)
                    if dist < dist_min:
                        dist_min = dist
                        match_num = j
            if groups[match_num] == -1:#新建一个类
                    node = Node(now[i].point_num + now[match_num].point_num)
                    next.append(node)
                    groups[i] = group_count
                    groups[match_num] = group_count
                    group_count += 1
            else:#归入旧类
                    next[groups[match_num]].point_num += now[i].point_num
                    groups[i] = groups[match_num]
    for i in next:
        i.terrianinfo()
    return Hierarchical_Clusteringluster(next)

#主程序
a ,label = preprocession()
test = []
for i in range(a.shape[0]):#初始化每个点为一个类
    node = Node([i])
    node.terrianinfo()
    test.append(node)
label_fore = Hierarchical_Clusteringluster(test)
np.savetxt("2016300030060.txt", label_fore.T)#打印到文件
print("nmi: " + str(metrics.normalized_mutual_info_score(label_fore,label)))#打印nmi
import math
import numpy as np
from sklearn import metrics
import random

class Node(object):
    def __init__(self, point_num):
        self.cluster_v = np.ndarray
        self.point_num = point_num

    def terrianinfo(self):#计算每个类的边界信息,采用均值来作为类中心点
        length = len(self.point_num)
        sum_point = a[self.point_num[0]]
        for i in range(1, length):
            sum_point += a[self.point_num[i]]
        self.cluster_v = sum_point / length

def variance(a):
    for i in range(a.shape[1]):
        tmp = a[:, i]
        var = np.var(tmp)
        print("var" + str(i) + ": " + str(var))
    

def preprocession():#预处理
    a_label = np.loadtxt('breast.txt')
    label = a_label[:, 9].astype(np.int32)
    a = np.delete(a_label, -1, axis=1)
    return a, label
    
def caldist(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def core(now, iter):
    if iter == 0:
       ret = np.zeros(699, dtype = np.int)
       for i in range(len(now)):
            #print("第%d个类包含:" %i)
            #print(now[i].point_num)
            for j in now[i].point_num:
                ret[j] = (i+1) * 2
       return ret
    next = []
    length = len(now)
    flag = [False for i in range(length)]
    for i in range(length):
        if flag[i] is False:
            dist_min = 100000
            match_num = 0
            for j in range(i+1, length):
                if flag[j] is False:
                    dist = caldist(now[i].cluster_v, now[j].cluster_v)#计算两个类的相似度
                    if dist < dist_min:
                        dist_min = dist
                        match_num = j
            #将两个子类合并为新的类
            flag[i] = True
            flag[match_num] = True
            if not match_num == 0:
              node_new = Node(now[i].point_num + now[match_num].point_num)
              node_new.terrianinfo()
              next.append(node_new)
            else:
              node_new = Node(now[i].point_num)
              node_new.terrianinfo()
              next.append(node_new)
    return core(next, iter-1)
    
    
a ,label = preprocession()
a[:, 0] = a[:, 0] * 7.9 * 7.9 * 7.9
a[:, 1] = a[:, 1] * 9.3 * 9.3 * 9.3
a[:, 2] = a[:, 2] * 8.8 * 8.8 * 8.8
a[:, 3] = a[:, 3] * 8.1 * 8.1 * 8.1
a[:, 4] = a[:, 4] * 4.9 * 4.9 * 4.9
a[:, 5] = a[:, 5] * 13.1 * 13.1 * 13.1
a[:, 6] = a[:, 6] * 5.9 * 5.9 * 5.9
a[:, 7] = a[:, 7] * 9.3 * 9.3 * 9.3
a[:, 8] = a[:, 8] * 2.9 * 2.9 * 2.9
a = a / 125
test = []
print(label)
count = 0

for i in range(a.shape[0]):
    node = Node([i])
    node.terrianinfo()
    test.append(node)
label_fore = core(test, 9)
print(label_fore)
print(metrics.normalized_mutual_info_score(label_fore,label))



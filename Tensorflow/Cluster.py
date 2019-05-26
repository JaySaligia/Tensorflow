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
    
def std(a):
    print(np.std(a, axis=0))



def preprocession():#预处理
    a_label = np.loadtxt('breast.txt')
    label = a_label[:, 9].astype(np.int32)
    a = np.delete(a_label, -1, axis=1)
    std(a)
    b = np.zeros((699,9), dtype = float)
    b[:, 0] = a[:, 0] 
    b[:, 1] = a[:, 1]
    b[:, 2] = a[:, 2]
    b[:, 3] = a[:, 3]
    b[:, 4] = a[:, 4] * 0.9
    b[:, 5] = a[:, 5]
    b[:, 6] = a[:, 6]
    b[:, 7] = a[:, 7]
    b[:, 8] = a[:, 8] * 0.5

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
    
    #a = np.loadtxt('lower.txt')
    return b, label
    
def caldist(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def caldist2(node0, node1):
    ret = 0
    for i in range(len(node0.point_num)):
        for j in range(len(node1.point_num)):
            dist = np.linalg.norm(a[node0.point_num[i]] - a[node1.point_num[j]])
            ret += dist
    return ret / (len(node0.point_num) * len(node1.point_num))

def core(now, iter):#最简单的方法
    print("iter: " + str(iter))
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
                    #计算两个类的相似度
                    dist = caldist(now[i].cluster_v, now[j].cluster_v)
                    #dist = caldist2(now[i], now[j])
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
    
def core2(now):
    length = len(now)
    next = []
    for i in range(length):
        next.append(now[i])
    if length == 2:
        ret = np.zeros(699, dtype = np.int)
        for i in range(length):
            print("No." + str(i) + " cluster:")
            print(now[i].point_num)
            for j in now[i].point_num:
                if i == 0:
                    ret[j] = 4
                else:
                    ret[j] = 2
        return ret
    print("having: " + str(length) + " clusters")
    dist_min = 100000
    son_0 = 0
    son_1 = 0
    for i in range(length):
        for j in range(i+1, length):
            dist = caldist(now[i].cluster_v, now[j].cluster_v)
            if dist < dist_min:
                son_0 = i
                son_1 = j
                dist_min = dist
    node_new = Node(now[son_0].point_num + now[son_1].point_num)
    node_new.terrianinfo()
    next.pop(son_0)
    next.pop(son_1-1)
    next.append(node_new)
    return core2(next)

def core3(now):#update dump arg
    length = len(now)
    print("having " + str(length) + " clusters")
    if length == 1:
        return np.zeros(699, dtype = np.int)
    if length == 2:
       ret = np.zeros(699, dtype = np.int)
       for i in range(length):
            #print("第%d个类包含:" %i)
            #print(now[i].point_num)
            for j in now[i].point_num:
                ret[j] = (i+1) * 2
            print(now[i].point_num)
       return ret
    
    
    next = []
    groups = [-1 for i in range(length)]
    group_count = 0
    dist_count = 0
    dist_total = 0
    #for i in range(length):
    #    for j in range(i+1, length):
    #        dist = caldist(now[i].cluster_v, now[j].cluster_v)
    #        dist_count += dist
    #        dist_total += 1
    #avg = dist_count / dist_total
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
            #if dist_min > avg:
            #        node = Node(now[i].point_num)
            #        next.append(node)
            #        groups[i] = group_count
            #        group_count += 1
            #else:
            if groups[match_num] == -1:
                    node = Node(now[i].point_num + now[match_num].point_num)
                    next.append(node)
                    groups[i] = group_count
                    groups[match_num] = group_count
                    group_count += 1
            else:
                    next[groups[match_num]].point_num += now[i].point_num
                    groups[i] = groups[match_num]
    for i in next:
        i.terrianinfo()
    #print("dist_mean: " + str(dist_total / dist_count))
    return core3(next)


a ,label = preprocession()
test = []
print(label)
count = 0
#std(a)
for i in range(a.shape[0]):
    node = Node([i])
    node.terrianinfo()
    test.append(node)
#core2(test)
label_fore = core3(test)
print(label_fore)

count_fault = 0
t = np.zeros(699, dtype=int)
for i in range(len(label_fore)):
    if label_fore[i] == label[i]:
        t[i] = 1
    else:
        t[i] = 0
        count_fault += 1

print(t)
print(count_fault)
print(metrics.normalized_mutual_info_score(label_fore,label))



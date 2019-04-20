#HMM前向算法
import numpy as np

N = 3
M = 3
T = 8

A = np.zeros((N, N), dtype=float)#状态转移概率矩阵
B = np.zeros((N, M), dtype=float)#观测概率矩阵
P = np.zeros((N, ), dtype=float)#初始时状态概率分布
O = np.zeros((T, ), dtype=int)#观测序列

a = np.zeros((T, N), dtype=float)

A = np.array([[0.8, 0.3, 0.4],[0.1, 0.4, 0.2],[0.1, 0.3, 0.4]])
B = np.array([[0.8, 0.2, 0.1], [0.1, 0.6, 0.3], [0.1, 0.2, 0.6]])
P = np.array([0.64, 0.19, 0.27])
O = np.array([0, 0, 1, 2, 1, 2, 1, 0])
def init():
    for i in range(N):
        a[0][i] = P[i] * B[i][O[0]]

def front_sum(t, q):
    total = 0
    for i in range(N):
        total += a[t-1][i] * A[i][q] 
    return total


def cal():
    for t in range(1, T):
        for q in range(N):
            a[t][q] = front_sum(t, q) * B[q][O[t]]
    total = 0
    for i in range(N):
        total += a[t][i]
    return total



#HMM维特比算法
import numpy as np

N = 3
M = 3
T = 8

A = np.zeros((N, N), dtype=float)#状态转移概率矩阵
B = np.zeros((N, M), dtype=float)#观测概率矩阵
P = np.zeros((N, ), dtype=float)#初始时状态概率分布
O = np.zeros((T, ), dtype=int)#观测序列
F = np.zeros((T, N), dtype=int)#Φ矩阵
d = np.zeros((T, N), dtype=float)#δ矩阵
result = []#输出矩阵

A = np.array([[0.8, 0.3, 0.4],[0.1, 0.4, 0.2],[0.1, 0.3, 0.4]])
B = np.array([[0.8, 0.2, 0.1], [0.1, 0.5, 0.2], [0.1, 0.3, 0.7]])
P = np.array([0.64, 0.19, 0.27])
O = np.array([0, 0, 1, 2, 1, 2, 1, 0])
def init():
    for i in range(N):
        d[0][i] = P[i] * B[i][O[0]]
        F[0][i] = 0

def viterbi(t, q):
    max_val = 0#最大值
    max_route = 0#最大路径来源
    for i in range(N):
        tmp = d[t-1][i] * A[i][q]
        if tmp > max_val:
            max_val = tmp
            max_route = i
    F[t][q] = max_route
    return max_val


def calculate():
    #计算Φ矩阵和δ矩阵
    for t in range(1, T):
        for q in range(N):
            d[t][q] = viterbi(t,q) * B[q][O[t]]
    max_route = 0
    max_val = 0
    for i in range(N):
        tmp = d[t][i]
        if tmp > max_val:
            max_val = tmp
            max_route = i
    result.append(str(i))

    i_s = i
    for i in range(T-1):
        tmp = F[T-1-i][i_s]
        result.append(str(tmp))
        i_s = tmp
    print("热,热,冷,湿,冷,湿,冷,热")
    return ",".join(result)[::-1].replace("0", "晴").replace("1", "阴").replace("2", "雨")

init()
print(calculate())

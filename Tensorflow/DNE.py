#DNE降维
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
filename = "c:\\tfmodels\\dataset2_data_mining_course.csv"


def make_matrix(filename):
    matrix = np.loadtxt(open(filename,"rb"), delimiter=",", skiprows=0)
    return matrix

x_raw = make_matrix(filename)
d = int(input("输入维度"))
groundtruth = []
for i in range(100):
    groundtruth.append("A")
for i in range(100, 300):
    groundtruth.append("B")
for i in range(300, 500):
    groundtruth.append("C")

def judge(a, b):
    val = 7#这个值是调参比较好的结果
    sum = (x_raw[a][0]-x_raw[b][0])**2 + (x_raw[a][1]-x_raw[b][1])**2 + (x_raw[a][2]-x_raw[b][2])**2
    if sum < val and groundtruth[a] == groundtruth[b]:
        return 1
    elif sum < val and groundtruth[a] != groundtruth[b]:
        return -1
    else:
        return 0

def dne(x_raw, d):#x_raw为输入矩阵（n*p），d为维数
    with tf.name_scope("pca"):
        #得到矩阵的大小
        x_in = tf.convert_to_tensor(x_raw)
        x_in = tf.cast(x_in, tf.float32)
        #创建邻接矩阵（这里没有采用k近邻方法，因为复杂度太高，而使用的是ξ方法）
        W = np.zeros((500, 500))
        D = np.zeros((500, 500))
        for i in range(500):
            for j in range(500):
                W[i][j] = judge(i, j)
        for i in range(500):
            count = 0
            for j in range(500):
                count += W[j][i]
            D[i][i] = count
        L = D - W
        l = tf.convert_to_tensor(L)
        l = tf.cast(l, tf.float32)
        tmp = tf.matmul(x_in, l, transpose_a=True)
        cov = tf.matmul(tmp, x_in)
        #特征值分解
        e, v = tf.linalg.eigh(cov)
        #对得到的特征值中取前d个最大的
        e_index = tf.math.top_k(e, sorted=True, k =d)[1]
        #取前d个最大特征向量
        v_dne = tf.gather(v, indices=e_index)
        #得到dne结果矩阵
        x_dne = tf.matmul(x_in, v_dne, transpose_b=True)        
        sess = tf.Session()
        #转为numpy矩阵
        x_dne_np = x_dne.eval(session=sess)
    return x_dne_np

result = dne(x_raw, d)

def showmodel():
    if (d == 2):#二维分布  
        result_1 = result[0:100,:]
        result_2 = result[100:300,:]
        result_3 = result[300:500,:]
        plt.scatter(result_1[:,0], result_1[:,1], c='blue')
        plt.scatter(result_2[:,0], result_2[:,1], c='orange')
        plt.scatter(result_3[:,0], result_3[:,1], c='red')
        plt.show()
    elif (d == 1):#一维分布
        result_1 = result[0:100,:]
        result_2 = result[100:300,:]
        result_3 = result[300:500,:]
        plt.scatter(result_1[:,0], np.zeros(100), c='blue')
        plt.scatter(result_2[:,0], np.zeros(200), c='orange')
        plt.scatter(result_3[:,0], np.zeros(200), c='red')
        plt.show()
    else:#原始3d分布
        result_1 = x_raw[0:100,0:3]
        result_2 = x_raw[100:300,0:3]
        result_3 = x_raw[300:500,0:3]
        model = plt.subplot(111, projection='3d')
        plt.scatter(result_1[:,0], result_1[:,1], result_1[:,2], c='blue')
        plt.scatter(result_2[:,0], result_2[:,1], result_2[:,2], c='orange')
        plt.scatter(result_3[:,0], result_3[:,1], result_3[:,2], c='red')
        plt.show()


showmodel()

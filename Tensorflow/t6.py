import tensorflow as tf
import numpy as np
#PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
filename = "c:\\tfmodels\\dataset2_data_mining_course.csv"

def make_matrix(filename):
    matrix = np.loadtxt(open(filename,"rb"), delimiter=",", skiprows=0)
    return matrix

def pca(x_raw, d):#x_in为输入矩阵（n*p），d为维数
    with tf.name_scope("pca"):
        #得到矩阵的大小
        x_in = tf.convert_to_tensor(x_raw)
        x_in = tf.cast(x_in, tf.float32)
        n, m= tf.to_float(x_in.get_shape()[0]),tf.to_int32(x_in.get_shape()[1])
        #计算出输入矩阵每行的平均值
        mean = tf.reduce_mean(x_in, axis=1)
        #reshape把行矩阵变为向量，然后对每一项取去中心化
        x_mean = x_in - tf.reshape(mean,(-1,1))
        
        #计算协方差矩阵
        cov = tf.matmul(x_mean, x_mean, transpose_a=True)/(n - 1)
        #特征值分解
        e, v = tf.linalg.eigh(cov)
        #对得到的特征值中取前d个最大的
        e_index = tf.math.top_k(e, sorted=True, k =d)[1]
        #取前d个最大特征向量
        v_pca = tf.gather(v, indices=e_index)
        #得到pca结果矩阵
        x_pca = tf.matmul(x_mean, v_pca, transpose_b=True)        
        sess = tf.Session()
        #转为numpy矩阵
        x_pca_np = x_pca.eval(session=sess)
    #return sess.run(x_pca)
    return x_pca_np

x_raw = make_matrix(filename)
d = int(input("输入维度"))
result = pca(x_raw, d)

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
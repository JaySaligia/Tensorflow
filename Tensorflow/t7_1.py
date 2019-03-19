#lda_2(关于均值有点问题还没有弄清楚)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
filename = "c:\\tfmodels\\dataset2_data_mining_course.csv"

def make_matrix(filename):
    matrix = np.loadtxt(open(filename,"rb"), delimiter=",", skiprows=0)
    return matrix

def lda(x_raw, x_raw_1, x_raw_2, x_raw_3 ,d):#x_in为输入矩阵（n*p），d为维数
    with tf.name_scope("lda"):
        #得到矩阵的大小
        x_in = tf.convert_to_tensor(x_raw)
        x_in = tf.cast(x_in, tf.float32)
        x_in_1 = tf.convert_to_tensor(x_raw_1)
        x_in_1 = tf.cast(x_in_1, tf.float32)#A类
        x_in_2 = tf.convert_to_tensor(x_raw_2)
        x_in_2 = tf.cast(x_in_2, tf.float32)#B类
        x_in_3 = tf.convert_to_tensor(x_raw_3)
        x_in_3 = tf.cast(x_in_3, tf.float32)#C类

        n = tf.to_float(x_in.get_shape()[0]),tf.to_int32(x_in.get_shape()[1])
        #总样本均值和各类均值
        mean = tf.reduce_mean(x_in, axis=1)
        mean_1 = tf.reduce_mean(x_in_1, axis=1)
        mean_2 = tf.reduce_mean(x_in_2, axis=1)
        mean_3 = tf.reduce_mean(x_in_3, axis=1)
        mean_total = x_in - tf.reshape(mean,(-1,1))
        mean_1_tmp = x_in_1 - tf.reshape(mean_1, (-1,1))
        mean_2_tmp = x_in_2 - tf.reshape(mean_2, (-1,1))
        mean_3_tmp = x_in_3 - tf.reshape(mean_3, (-1,1))
        mean_diff = tf.concat([mean_1_tmp, mean_2_tmp], 0)
        mean_diff = tf.concat([mean_diff, mean_3_tmp], 0)
        #计算类内散度矩阵Sw
        Sw_mean = x_in - mean_diff
        Sw = tf.matmul(Sw_mean, Sw_mean, transpose_a=True)
        #计算类间散度矩阵Sb
        St_mean = x_in - mean_total
        St = tf.matmul(St_mean, St_mean, transpose_a=True)#全局散度
        Sb = St - Sw
        cov = tf.matmul(tf.matrix_inverse(Sw), Sb)
        #特征值分解
        e, v = tf.linalg.eigh(cov)
        #对得到的特征值中取前d个最大的
        e_index = tf.math.top_k(e, sorted=True, k =d)[1]
        #取前d个最大特征向量
        v_lda = tf.gather(v, indices=e_index)
        #得到pca结果矩阵
        x_lda = tf.matmul(x_in, v_lda, transpose_b=True)        
        sess = tf.Session()
        #转为numpy矩阵
        x_lda_np = x_lda.eval(session=sess)
    #return sess.run(x_pca)
    return x_lda_np

x_raw = make_matrix(filename)
d = int(input("输入维度"))
x_raw1 = x_raw[0:100,:]
x_raw2 = x_raw[100:300,:]
x_raw3 = x_raw[300:500,:]
result = lda(x_raw, x_raw1, x_raw2, x_raw3, d)

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


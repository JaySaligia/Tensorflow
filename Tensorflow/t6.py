import tensorflow as tf
#PCA
def pca(x_in, d):#x_in为输入矩阵（n*p），d为维数
    with tf.name_scope("pca"):
        #得到矩阵的大小
        n = tf.to_float(x_in.get_shape()[0])
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
    return x_pca
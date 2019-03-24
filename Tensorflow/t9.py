#Inception Net
import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev : tf.truncated_normal_initializer(0.0, stddev)#匿名函数

def inception_v3_arg_scope(
    weight_decay = 0.00004,
    seddev = 0.1,
    batch_norm_var_collection = 'moving_vars'
    ):
    tbatch_norm_paras = {
        'decay:': 0.9997,
        'epsilon': 0.001,
        'update_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta': None,
            'gamma': None,
            'moving_mean':[batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
            }
        }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params) as sc:
           return sc

def inception_v3_base(inputs, scope=None):
    endpoints = {}
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
            #五层卷积层和两层池化层;five Convolutional layers and two Pooling layers
            net = slim.conv2d(inputs, 32, [3,3], stride=2, scope='Conv2d_la_3x3')
            net = slim.conv2d(net, 32, [3,3], scope='Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3,3], padding='SAME', scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1,1], scope='Conv2d_3b_1x1')
            net = slim.conv2d(net, 192, [3,3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_5a_3x3')
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            #第一个Inception模块组包含3个Inception Module;The first group of Inception modules includes 3 modules
            #第一个模块(Mixed_5b)，包含4个分支;The first module(Mixed_5b) includes 4 branches
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(brahch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            #第二个模块(Mixed_5c)，包含4个分支;The second module(Mixed_5c) includes 4 branches
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(brahch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            #第三个模块(Mixed_5d)，包含4个分支;The second module(Mixed_5d) includes 4 branches
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(brahch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

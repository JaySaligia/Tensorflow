#Inception Net
from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import graph_util
slim = tf.contrib.slim
trunc_normal = lambda stddev : tf.truncated_normal_initializer(0.0, stddev)#匿名函数
batch_size = 32
num_batches = 100
max_step = 3000
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
def inception_v3_arg_scope(
    weight_decay = 0.00004,
    stddev = 0.1,
    batch_norm_var_collection = 'moving_vars'
    ):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
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
    end_points = {}
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
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
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
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
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
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            #第二个Inception模块组包含5个Inception Module;The second group of Inception modules includes 5 modules
            #第一个模块(Mixed_6a)，包含三个分支;The first module(Mixed_6a) includes 3 branches
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            #第二个模块(Mixed_6b)，包含四个分支;The second module(Mixed_6b) includes 4 branches
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    brahcn_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            #第三个模块(Mixed_6c)，包含四个分支;The third module(Mixed_6c) includes 4 branches
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    brahcn_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            #第四个模块(Mixed_6d)，包含四个分支;The forth module(Mixed_6d) includes 4 branches
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    brahcn_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            #第五个模块(Mixed_6e)，包含四个分支;The fifth module(Mixed_6e) includes 4 branches
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    brahcn_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points['Mixed_6e'] = net
            #第三个Inception模块组包含3个Inception Module;The third group of Inception modules includes 3 modules
            #第一个模块(Mixed_7a)，包含三个分支;The first module(Mixed_7a) includes 3 branches
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            #第二个模块(Mixed_7b)，包含四个分支;The second module(Mixed_7b) includes 4 branches
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            #第三个模块(Mixed_7c)，包含四个分支;The third module(Mixed_7c) includes 4 branches
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            return net, end_points

def inception_v3(inputs,
                 num_classes=28,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v3_base(inputs, scope=scope)
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                        aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
                        aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
                        aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01), padding='VALID', scope='Conv2d_2a_5x5')
                        aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
                        if spatial_squeeze:
                            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                        end_points['AuxLogits'] = aux_logits
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                end_points['Logits'] = logits
                end_points['Predicitons'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

#batch_size = 32
#height, width = 299,299
#inputs = tf.random_uniform((batch_size, height, width, 3))
#with slim.arg_scope(inception_v3_arg_scope()):
#    logits, end_points = inception_v3(inputs, is_training=False)

#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
#num_batches = 100
#time_tensorflow_run(sess, logits, "Forward")
def losses(logits, labels):
    labels = tf.cast(labels, tf.int64)
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def get_accuracy(logits, label):
    current = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), 'float')
    accuracy = tf.reduce_mean(current)
    return accuracy

def read_train_data():
    reader = tf.TFRecordReader()
    filename_train = tf.train.string_input_producer(["commodity_train.tfrecords"])
    _, serialized_example_test = reader.read(filename_train)
    features = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            }
        )

    img_train = features['image']
    images_train = tf.decode_raw(img_train, tf.uint8)
    images_train = tf.reshape(images_train, [299, 299, 3])
    labels_train = tf.cast(features['label'], tf.int64)
    labels_train = tf.cast(labels_train, tf.int64)
    labels_train = tf.one_hot(labels_train, 28)
    return images_train, labels_train
#读测试集
def read_test_data():
    reader = tf.TFRecordReader()
    filename_test = tf.train.string_input_producer(["commodity_test.tfrecords"])
    _, serialized_example_test = reader.read(filename_test)
    features = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            }
        )

    img_test = features['image']
    images_test = tf.decode_raw(img_test, tf.uint8)
    images_test = tf.reshape(images_test, [299, 299, 3])

    labels_test = tf.cast(features['label'], tf.int64)
    labels_test = tf.cast(labels_test, tf.int64)
    labels_test = tf.one_hot(labels_test, 28)
    return images_test, labels_test
def save_model(sess, step):
    MODEL_SAVE_PATH = ""
    MODEL_NAME = "model.ckpt"
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)

def train():
    x_train, y_train = read_train_data()
    x_test, y_test = read_test_data()
    x_batch_train, y_batch_train = tf.train.shuffle_batch([x_train, y_train], batch_size=batch_size, capacity=200,
                                                          min_after_dequeue=100, num_threads=3)
    x_batch_test, y_batch_test = tf.train.shuffle_batch([x_test, y_test], batch_size=batch_size, capacity=200,
                                                        min_after_dequeue=100, num_threads=3)
    x = tf.placeholder(tf.float32, shape=[None, 268203])#299*299*3
    y = tf.placeholder(tf.int64, shape=[None, 28])
    #x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    #y = tf.placeholder(tf.int32, [batch_size])
    images = tf.reshape(x, shape=[batch_size, 299, 299, 3])

    logits, end_points = inception_v3(images, is_training=True)
   
    getAccuracy = get_accuracy(logits, y)
    global_step = tf.Variable(0, name='global_step')
    loss = losses(logits, y)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    #top_k_op = tf.nn.in_top_k(logits, [batch_size], 1)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(max_step):
        start_time = time.time()
        images_train, label_train = sess.run([x_batch_train, y_batch_train])
        _images_train = np.reshape(images_train, [batch_size, 268203])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: _images_train,y: label_train})
        duration = time.time() - start_time
        if i % 10 == 0:
            accuracy = sess.run(getAccuracy, feed_dict={x: _images_train, y: label_train})
            example_per_sec = batch_size / duration
            sec_per_batch = float(duration)
           
            format_str = ('step %d,loss = %.2f, accuracy = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (i, loss_value, accuracy, example_per_sec, sec_per_batch ))
    for i in range(10):
        images_test, label_test = sess.run([x_batch_test, y_batch_test])
        _images_test = np.reshape(images_test, [batch_size, 268203])
        accuracy_test = sess.run(getAccuracy, feed_dict={x: _images_test, y: label_test})
        print("test accuracy: %g" % accuracy_test)
    


    #save_model(sess, max_step)
    tf.saved_model.simple_save(session=sess,
    export_dir="pb",
    inputs={"x": x, "y": y},
    outputs={"logits": logits})
    coord.request_stop()
    coord.join(threads)


train()
            

 





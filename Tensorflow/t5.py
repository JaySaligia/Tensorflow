#AlexNet(做商品识别)
from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np

batch_size = 32
num_batches = 100
max_step = 5000

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

#增加L2正则化的loss
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def inference(images):#接受原始输入为224*224*3；256*256随机截取
    parameters = []
    with tf.name_scope('conv1')as scope:#第一层，卷积核尺寸11*11，步长为4，最大池化层尺寸3*3，步长为2
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2')as scope:#第二层，卷积核尺寸5*5，步长为1，最大池化层尺寸3*3，步长为2
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)
    
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],  padding='VALID', name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:#第三层，卷积核尺寸为3*3，数量为384,无池化层
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:#第四层，卷积核数量降为256，无池化层
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    with tf.name_scope('conv5') as scope:#第五层，如第四层
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
    
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    return fully_connected_layer(pool5)

def fully_connected_layer(pool):#全连接层，共三层
    reshape = tf.reshape(pool, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
    bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
    local1 = tf.nn.relu(tf.matmul(reshape, weight1) + bias1)

    weight2 = variable_with_weight_loss(shape=[384, 196], stddev=0.04, w1=0.004)
    bias2 = tf.Variable(tf.constant(0.1, shape=[196]))
    local2 = tf.nn.relu(tf.matmul(local1, weight2) + bias2)

    weight3 = variable_with_weight_loss(shape=[196, 28], stddev=1/192.0, w1=0.0)
    bias3 = tf.Variable(tf.constant(0.0, shape=[28]))
    logits = tf.add(tf.matmul(local2, weight3), bias3)

    return logits

def losses(logits, labels):
    labels = tf.cast(labels, tf.int64)
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

#以下代码学习自https://github.com/wmpscc/TensorflowBaseDemo/blob/master/CNN_train.py
#读训练集
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
    images_train = tf.reshape(images_train, [224, 224, 3])
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
    images_test = tf.reshape(images_test, [224, 224, 3])
    labels_test = tf.cast(features['label'], tf.int64)
    labels_test = tf.cast(labels_test, tf.int64)
    labels_test = tf.one_hot(labels_test, 28)
    return images_test, labels_test

def train():
    x_train, y_train = read_train_data()
    x_test, y_test = read_test_data()
    x_batch_train, y_batch_train = tf.train.shuffle_batch([x_train, y_train], batch_size=BATCH_SIZE, capacity=200,
                                                          min_after_dequeue=100, num_threads=3)
    x_batch_test, y_batch_test = tf.train.shuffle_batch([x_test, y_test], batch_size=BATCH_SIZE, capacity=200,
                                                        min_after_dequeue=100, num_threads=3)
    x = tf.placeholder(tf.float32, shape=[None, 150528])#224*224*3
    y = tf.placeholder(tf.int64, shape=[None, 28])

    images = tf.reshape(x, shape=[batch_size, 224, 224, 3])

    logits = inference(images)
    getAccuracy = get_accuracy(logits, y)
    global_step = tf.Variable(0, name='global_step')
    loss = losses(logits, y)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    top_k_op = tf.nn.in_top_k(logits, y, 1)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(max_step):
        start_time = time.time()
        images_train, label_train = sess.run([x_batch_train, y_batch_train])
        _images_train = np.reshape(images_train, [batch_size, 150528])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: _images_train,y: label_train})
        duration = time.time() - start_time
        if i % 10 == 0:
            example_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            format_str = ('step %d,loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (i, loss_value, examples_per_sec, sec_per_batch))
    
    #images_test, label_test = sess.run([x_batch_test, y_batch_test])
    #_images_test = np.reshape(images_test, [batch_size, 150528])
    #accuracy_test = sess.run(getAccuracy, feed_dict={x: _images_test, y: label_test})
    #print("test accuracy: %g" % accuracy_test)
    num_examples= 200
    import math
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([x_batch_test, y_batch_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
        true_count += np.sum(predictions)
        step += 1

    precision = ture_count/total_sample_count
    print('precision @ 1 = %.3f' % precision)


    #save_model(sess, i)
    coord.request_stop()
    coord.join(threads)

train()
            


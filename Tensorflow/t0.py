#softmax regression识别minst
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
minst = input_data.read_data_sets("MNIST_data/", one_hot = True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+ b)#y = softmax(Wx + b)
y_ = tf.placeholder(tf.float32,[None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))#loss func
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

print("begin\n")
for i in range(1000):
    batch_xs, batch_ys = minst.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: minst.test.images, y_: minst.test.labels}))


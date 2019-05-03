import tensorflow as tf
import numpy as np

train_x1 = np.linspace(-10, 10, 300)
# train_x2 = np.linspace(-10, 10, 300)
# train_x3 = np.linspace(-10, 10, 300)

train_y = train_x1

X1 = tf.placeholder('float32')
# X2 = tf.placeholder('float32')
# X3 = tf.placeholder('float32')
Y = tf.placeholder('float32')

W1 = tf.Variable(tf.random_normal([1]))
# W2 = tf.Variable(tf.random_normal([1]))
# W3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.zeros([1]))

z = tf.multiply(W1, X1) + b

cost = tf.reduce_mean(tf.square(z-Y))

learning_rate = 0.001

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

train_epoch = 100
display_step = 3

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(train_epoch):
        for (x1, y) in zip(train_x1, train_y):
            sess.run(optimizer, feed_dict={X1:x1, Y:y})

    print(sess.run(W1),sess.run(b))


import tensorflow as tf
import numpy as np

def read_in(file_name):
    text = open(file_name, mode='r')
    read_temp = text.readlines()
    case_num = 0

    case_num = len(read_temp)

    #初始化数组
    X_ip = np.zeros([case_num, 3])
    Y_hat = np.zeros([case_num, 1])

    i = 0
    while read_temp[i]:
        [X_ip[i][0],X_ip[i][1],X_ip[i][2],Y_hat[i][0],_,_,_,_,_] = read_temp[i].split(",")
        i += 1
        if i == case_num:
            break

    return X_ip, Y_hat, case_num

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([3, 8]))
b1 = tf.Variable(tf.zeros([8]))
z1 = tf.add(tf.matmul(x, W1), b1)
a1 = tf.nn.sigmoid(z1)
W2 = tf.Variable(tf.random_normal([8, 6]))
b2 = tf.Variable(tf.zeros([6]))
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = tf.nn.sigmoid(z2)
W3 = tf.Variable(tf.random_normal([6, 1]))
b3 = tf.Variable(tf.zeros([1]))
z3 = tf.add(tf.matmul(a2, W3), b3)


cost = tf.reduce_sum(tf.square(z2 - y))

learning_rate = 0.03

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.2, beta1=0.9, beta2=0.99, epsilon=1e-08,
#                                     use_locking=False, name='Adam').minimize(cost)

training_epochs = 50
display_step = 1
X_ip, Y_ip, case_num = read_in("D:\Group_Zhang\AI_float\ip_new.txt")
print(Y_ip)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        cost_value = 0
        for i in range(case_num):
            _, cost_value_temp = sess.run([optimizer, cost], feed_dict={x:[X_ip[i]], y:[Y_ip[i]]})
            cost_value += cost_value_temp

        if (epoch + 1) %display_step == 0:
            print('Epoch:', '%04d'%(epoch+1), 'Cost=', str(cost_value))

    print('finished')
    print(sess.run(z2, feed_dict={x:[[1,2,3]]}))

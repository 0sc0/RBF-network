import numpy as np
import math
import tensorflow as tf

#读取文本
def read_in(file_name):
    text = open(file_name, mode='r')
    read_temp = text.readlines()
    case_num = 0

    case_num = len(read_temp)

    #初始化数组
    X_ip = np.zeros([case_num, 3])
    Y_hat = np.zeros([case_num, 6])

    i = 0
    while read_temp[i]:
        [X_ip[i][0],X_ip[i][1],X_ip[i][2],
        Y_hat[i][0],Y_hat[i][1],Y_hat[i][2],
        Y_hat[i][3],Y_hat[i][4],Y_hat[i][5],] = read_temp[i].split(",")
        i += 1
        if i == case_num:
            break

    return X_ip.T, Y_hat.T, case_num

#初始化神经网络，决定网络形式
def initialize_parameters():
    W1 = tf.get_variable("W1", [8,3], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [8,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [8, 8], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [8, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 8], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    #写入字典
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters

#创建placeholder
def create_placeholder(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x,1])
    Y = tf.placeholder(tf.float32, shape=[n_y,1])

    return X, Y

#正向传播
def forward_propagation(X_ip, parameters):
    X = tf.placeholder("float32")

    W1 = parameters["W1"]
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    ### START CODE HERE ### (approx. 5 lines) # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)   # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)               # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)    # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)               # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)     # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    return Z3

#计算罚函数
def compute_cost(Z3, Y):
    cost = tf.reduce_sum(tf.square(Z3 - Y))
    return cost

#组合模型
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,num_epochs = 1500):

    (m, n_x) = X_train.shape  # (n_x: input size, m : number of examples in
    n_y = Y_train.shape[1]  # n_y : output size
    costs = []  # To keep track of the cost

    X, Y = create_placeholder(n_x, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    print(Z3.shape)
    print(Y.shape)

    cost = compute_cost(Z3, Y)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)

    init = tf.global_variables_initializer()
    for epoch in range(num_epochs):
        with tf.Session() as sess:
            sess.run(init)

            sess.run(optimizer, feed_dict={X : X_train, Y : Y_train})
            # Z3 = sess.run(Z3, feed_dict={})
            # cost = sess.run(cost, feed_dict={})



#执行

X_ip, Y_ip, case_num = read_in("D:\Group_Zhang\AI_float\ip_test.txt")
print(X_ip)
print(X_ip[:1])
print(Y_ip)
print(case_num)

# with tf.Session() as sess:
#     pra = initialize_parameters()
#     X, Y = create_placeholder(3, 6)
#     Z3 = forward_propagation(X, pra)
#     print(Z3)
# X_train, Y_train,num = read_in("D:\Group_Zhang\AI_float\ip_train.txt")
# X_test, Y_test, num_t = read_in("D:\Group_Zhang\AI_float\ip_test.txt")
#
#
# model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,num_epochs = 1500)
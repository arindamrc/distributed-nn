'''
Created on Dec 12, 2016

@author: arc
'''
import tensorflow as tf
import numpy as np
import inputs.xor

N_count = 2

indim = 2
outdim = 1
shape1 = [indim, N_count]
shape2 = [N_count, outdim]
X = tf.placeholder(tf.float32, [None, indim], name="X_input")
Y = tf.placeholder(tf.float32, [None, outdim], name="Y_input")
wh1 = tf.Variable(tf.random_uniform(shape1, -10, 10), name="wh1")
wh2 = tf.Variable(tf.random_uniform(shape2, -10, 10), name="wh2")
Bias1 = tf.Variable(tf.zeros([N_count]), name = "Bias1")
Bias2 = tf.Variable(tf.zeros([outdim]), name = "Bias2")

with tf.name_scope("l1") as scope:
    h = tf.sigmoid(tf.matmul(X, wh1) + Bias1)
    
with tf.name_scope("l2") as scope:
    predict_op = tf.sigmoid(tf.matmul(h, wh2) + Bias2)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_op, Y))
with tf.name_scope("cost") as scope:
    loss = tf.reduce_mean(( (Y * tf.log(predict_op)) + ((1 - Y) * tf.log(1.0 - predict_op)) ) * -1)
#         loss = -tf.reduce_sum(Y * tf.log(tf.nn.softmax(predict_op)))
with tf.name_scope("train") as scope:
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# XOR_X = [[0,0],[0,1],[1,0],[1,1]]
# XOR_Y = [[0],[1],[1],[0]]

session = tf.InteractiveSession()
tf.initialize_all_variables().run()

inputStream     = inputs.xor.XORProblem(nodes = 1, dim=2)

XOR_X = [] 
XOR_Y = []

# XOR_X = [[0,0],[0,1],[1,0],[1,1]]
# XOR_Y = [[0],[1],[1],[0]]

batch = 2

def debug_test(XOR_X, XOR_Y, l, p):
    a1 = session.run(wh1)
    a2 = session.run(wh2)
    print a1
    print a2
    print XOR_X
    print XOR_Y
    print l, p[0][0]

for i in xrange(batch * 100000):
    ex = inputStream.generate_example()
#     print ex
    x_val, y_val = ex
#     print x_val
    y_val = 0 if y_val == -1 else y_val
#     print y_val
    xlist = [0 if x==-1 else x for _, x in x_val]
#     print xlist
    x_val = np.array(xlist, dtype="int")
    x_val = np.reshape(x_val, (1, indim))[0]
    y_val = np.array([y_val], dtype="int")
    y_val = np.reshape(y_val, (1, outdim))[0]
#     print x_val
#     print y_val
#     t, l, p = session.run([train_op, loss, predict_op], feed_dict={X : x_val, Y : y_val})
#     if i%1000 == 0:
#         print l, p
    if i % batch == 0 and i != 0:
#         print XOR_X
#         print XOR_Y
#         l = session.run([loss], feed_dict={X : XOR_X, Y : XOR_Y})
#         l = session.run([predict_op], feed_dict={X : XOR_X, Y : XOR_Y})
        t, l, p = session.run([train_op, loss, predict_op], feed_dict={X : XOR_X, Y : XOR_Y})
        if i % 10000 == 0:
            debug_test(XOR_X, XOR_Y, l, p)
        XOR_X = []
        XOR_Y = []
    else:
        XOR_X.append(x_val.tolist())
        XOR_Y.append(y_val.tolist())


    
    
    
    
    
    


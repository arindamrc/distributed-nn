'''
Created on Dec 5, 2016

@author: arc
'''
import learning.model as model
import tensorflow as tf
import numpy as np
from numpy import dtype

N_count = 2

class DNNModel(model.Model):
    """
    A 2 hidden layer deep NN model.
    """
    def __init__(self, indim, outdim, name="dnn2"):
        self.indim = indim
        self.outdim = outdim
        self.name = name
        self.shape1 = [indim, N_count]
        self.shape2 = [N_count, outdim]
#         t1 = np.ones(self.shape1, dtype = "float32")
#         t2 = np.ones(self.shape2, dtype = "float32")
#         t1 = t1 * 0.5
#         t2 = t2 * 0.5
        self.wh1 = tf.Variable(tf.random_uniform(self.shape1, -1, 1), name="wh1")
#         self.wh1 = tf.Variable(t1, name="wh1")
        self.wh2 = tf.Variable(tf.random_uniform(self.shape2, -1, 1), name="wh2")
#         self.wh2 = tf.Variable(t2, name="wh2")
        self.b1 = tf.Variable(tf.zeros([N_count]), name="Bias1")
        self.b2 = tf.Variable(tf.zeros([outdim]), name="Bias2")
        self.modelType = "neural-network"
        self.X = tf.placeholder(tf.float32, [None, indim], name="X_input")
        self.Y = tf.placeholder(tf.float32, [None, outdim], name="Y_input")
        self._initTF()
        self.session = tf.InteractiveSession()
        tf.initialize_all_variables().run()

    def _initTF(self):
        with tf.name_scope("l1") as scope:
            self.h = tf.sigmoid(tf.matmul(self.X, self.wh1) + self.b1)
        with tf.name_scope("l2") as scope:
            self.predict_op = tf.sigmoid(tf.matmul(self.h, self.wh2) + self.b2)
#         self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict_op, self.Y))
        with tf.name_scope("cost") as scope:
#             self.loss = tf.reduce_mean(((self.Y * tf.log(self.predict_op)) + ((1 - self.Y) * tf.log(1.0 - self.predict_op))) * -1)
            self.loss = tf.reduce_mean(tf.square(self.Y - self.predict_op))
#         self.loss = -tf.reduce_sum(self.Y * tf.log(tf.nn.softmax(self.predict_op)))
        with tf.name_scope("train") as scope:
            self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
#         self.tfModel = tf.initialize_all_variables()
#         self.tfModel.run()

        
    def clone(self, other):
#         a = other.wh1.eval()
        a = other.session.run(other.wh1)
#         b = other.wh2.eval()
        b = other.session.run(other.wh2)
        self._assign(a, b)
        
    def distance(self, other):
#         a1 = self.wh1.eval()
        a1 = self.session.run(self.wh1)
#         b1 = other.wh1.eval()
        b1 = other.session.run(other.wh1)
#         a2 = self.wh2.eval()
        a2 = self.session.run(self.wh2)
#         b2 = other.wh2.eval()
        b2 = other.session.run(other.wh2)
#         d1 = a1 - b1
        n1 = np.linalg.norm(a1 - b1)
#         d2 = a2 - b2
        n2 = np.linalg.norm(a2 - b2)
        dist = np.average([n1, n2])
        print "distance: " + str(dist)
        return dist
    
    def norm(self):
#         a = self.wh1.eval()
        a = self.session.run(self.wh1)
#         b = self.wh2.eval()
        b = self.session.run(self.wh2)
        n = np.average([np.linalg.norm(a), np.linalg.norm(b)])
        print "norm: " + str(n)
        return n
    
    def train(self, example):
        x_val, y_val = example
#         x_val = [x for _, x in x_val]
        x_val = np.array(x_val, dtype="int")
        x_val = np.reshape(x_val, (1, self.indim))
        y_val = np.array([y_val], dtype="int")
        y_val = np.reshape(y_val, (1, self.outdim))
#         print x_val
#         print y_val
        _, l, p = self.session.run([self.train_op, self.loss, self.predict_op], feed_dict={self.X : x_val, self.Y : y_val})
        return p[0][0], l 
        
    def getPredictionScore(self, example):
        x_val, _ = example
        x_val = np.array([x for _, x in x_val])
        x_val = np.reshape(x_val, (1, self.indim))
        return self.session.run(self.predict_op, feed_dict={self.X : x_val})
    
    def getLoss(self, example):
        x_val, y_val = example
        x_val = np.array([x for _, x in x_val])
        x_val = np.reshape(x_val, (1, self.indim))
        y_val = np.array([y_val])
        y_val = np.reshape(y_val, (1, self.outdim))
        return self.session.run(self.train_op, feed_dict={self.X: x_val, self.Y : y_val})
    

    def _assign(self, a, b):
        op1 = self.wh1.assign(a)
        op2 = self.wh2.assign(b)
        self.session.run(op1)
        self.session.run(op2)
        t1 = self.session.run(self.wh1)
        t2 = self.session.run(self.wh2)

    def add(self, other):
#         a1 = self.wh1.eval()
        a1 = self.session.run(self.wh1)
#         b1 = other.wh1.eval()
        b1 = other.session.run(other.wh1)
#         a2 = self.wh2.eval()
        a2 = self.session.run(self.wh2)
#         b2 = other.wh2.eval()
        b2 = other.session.run(other.wh2)
        a1 = a1 + b1
        a2 = a2 + b2
        self._assign(a1, a2)
    
    def scalarMultiply(self, scalar):
#         a = self.wh1.eval()
        a = self.session.run(self.wh1)
#         b = self.wh2.eval()
        b = self.session.run(self.wh2)
        a = a * scalar
        b = b * scalar
        self._assign(a, b)
            
    def getInitParam(self):
        return {"indim" : self.indim, "outdim" : self.outdim, "name" : self.name}
    
    def getModelSize(self):
        return self.shape1[0] * self.shape1[1] + self.shape2[0] * self.shape2[1]
    
    def getModelParameters(self):
        return "None"
    
    def getModelIdentifier(self):
        return "NNModel"
        
    def getZeroModel(self, zero_weights = False):
        kwargs = self.getInitParam()
        zeroModel = self.__class__(**kwargs)
        if zero_weights:
            zeroModel.wh1 = tf.Variable(tf.zeros(self.shape1, dtype = tf.float32), name="wh1")
            zeroModel.wh2 = tf.Variable(tf.zeros(self.shape2, dtype = tf.float32), name="wh2")
            tf.initialize_all_variables().run()
        return zeroModel
    
    def fromRecord(self, record):
        return NotImplemented
    
    def print_state(self):
        a1 = self.session.run(self.wh1)
        a2 = self.session.run(self.wh2)
        print a1
        print a2
 
    
if __name__ == '__main__':
    v = tf.Variable(tf.zeros([10, 10]))
    

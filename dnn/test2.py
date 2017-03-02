'''
Created on Dec 12, 2016

@author: arc
'''
import learning.model as model
import tensorflow as tf
import numpy as np
import inputs.xor

N_count = 10

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
        self.wh1 = tf.Variable(tf.random_uniform(self.shape1, -1, 1), name="wh1")
        self.wh2 = tf.Variable(tf.random_uniform(self.shape2, -1, 1), name="wh2")
        self.modelType = "neural-network"
        self.session = tf.InteractiveSession()
        self.X = tf.placeholder("float", [1, indim])
        self.Y = tf.placeholder("float", [1, outdim])
        self._initTF()

    def _initTF(self):
        h = tf.sigmoid(tf.matmul(self.X, self.wh1))
        self.predict_op = tf.sigmoid(tf.matmul(h, self.wh2))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict_op, self.Y))
        self.loss = tf.reduce_mean(( (self.Y * tf.log(self.predict_op)) + ((1 - self.Y) * tf.log(1.0 - self.predict_op)) ) * -1)
        self.loss = -tf.reduce_sum(self.Y * tf.log(tf.nn.softmax(self.predict_op)))
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.tfModel = tf.initialize_all_variables()
        self.tfModel.run()

        
    def clone(self, other):
        a = other.wh1.eval()
        b = other.wh2.eval()
        self._assign(a, b)
        
    def distance(self, other):
        a1 = self.wh1.eval()
        b1 = other.wh1.eval()
        a2 = self.wh2.eval()
        b2 = other.wh2.eval()
        n1 = np.linalg.norm(a1 - b1)
        n2 = np.linalg.norm(a2 - b2)
        dist = np.average([n1, n2])
        print "distance: " + str(dist)
        return dist
    
    def norm(self):
        a = self.wh1.eval()
        b = self.wh2.eval()
        n = np.average([np.linalg.norm(a), np.linalg.norm(b)])
        print "norm: " + str(n)
        return n
    
    def train(self, example):
        x_val, y_val = example
        x_val = np.array([x for _, x in x_val], dtype="float")
        x_val = np.reshape(x_val, (1, self.indim))
        y_val = np.array([y_val], dtype="float")
        y_val = np.reshape(y_val, (1, self.outdim))
        self.session.run(self.train_op, feed_dict={self.X: x_val, self.Y : y_val})
        _, l, l = self.session.run([self.train_op, self.loss, self.predict_op], feed_dict={self.X : x_val, self.Y : y_val})
        return l[0][0], l 
        
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
        self.wh1.assign(a)
        self.wh2.assign(b)
        self.session.run(self.wh1)
        self.session.run(self.wh2)

    def add(self, other):
        a1 = self.wh1.eval()
        b1 = other.wh1.eval()
        a2 = self.wh2.eval()
        b2 = other.wh2.eval()
        a1 = a1 + b1
        a2 = a2 + b2
        self._assign(a1, a2)
    
    def scalarMultiply(self, scalar):
        a = self.wh1.eval()
        b = self.wh2.eval()
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
        
    def getZeroModel(self):
        kwargs = self.getInitParam()
        zeroModel = self.__class__(**kwargs)
        return zeroModel
    
    def fromRecord(self, record):
        return NotImplemented
    
    
if __name__ == '__main__':
    inputStream     = inputs.xor.XORProblem(nodes = 1, dim=10)
    model = DNNModel(10, 1)
    for _ in xrange(1000):
        example = inputStream.generate_example()
#         print example
        l, l = model.train(example)
        print l, l
        
        
        
        
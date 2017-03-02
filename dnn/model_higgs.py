'''
Created on Dec 5, 2016

@author: arc
'''
import learning.model as model
import tensorflow as tf
import numpy as np

class DNNModel(model.Model):
    """
    A 'n' layer deep NN model.
    indim : input dimension
    outdim : output dimension
    hidden_layers : hidden layer count
    neurons : number of neuron elements per hidden layer 
    """
    
    activations = {
        "relu" : tf.nn.relu,
        "tanh" : tf.nn.tanh,
        "sigmoid" : tf.nn.sigmoid
    }
    
    losses = {
        "l2" : lambda y, y_ : tf.square(y - y_),
        "xentropy" : lambda y, y_ : tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=tf.reshape(y_, [-1, 1]))
    }
    
    def __init__(self, indim, outdim, hidden_layers, neurons, batch_size, actfunc="relu", lossfunc="xentropy", learning_rate=0.1, name="dnn_higgs", zero_model=False):
        self.indim = indim
        self.outdim = outdim
        self.name = name
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.lr = learning_rate
        self._make_shape(zero_model)
        self.modelType = "neural-network"
        self.actfunc = actfunc
        self.lossfunc = lossfunc
        self.batch_size = batch_size
        self.parameters = "indim = %d, outdim = %d, hl = %d, nc = %d, lr = %d, activation = %s, loss = %s, name= %s" % (indim, outdim, hidden_layers, neurons, learning_rate, actfunc, lossfunc, name)
        self.X = tf.placeholder(tf.float32, [None, indim], name="X_input")
        self.Y = tf.placeholder(tf.float32, [None, outdim], name="Y_input")
        self._initTF()
        self.session = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        
        
    def _make_shape(self, zero_model=False):
        self.shapes = [None] * (self.hidden_layers + 1)
        self.shapes[0] = [self.indim, self.neurons]
        self.shapes[len(self.shapes) - 1] = [self.neurons, self.outdim]
        self.weights = [None] * (self.hidden_layers + 1)
        self.biases = [None] * (self.hidden_layers + 1) 
        self.biases[len(self.biases) - 1] = tf.Variable(tf.zeros([self.outdim]), name=("b%d" % (len(self.biases) - 1)))
        for i in range(1, self.hidden_layers):
            self.shapes[i] = [self.neurons, self.neurons]
        for j in range(len(self.shapes)):
            if not zero_model:
#                 self.weights[j] = tf.Variable(tf.random_uniform(self.shapes[j], -1, 1), name=("w%d" % j))
                self.weights[j] = tf.Variable(tf.random_normal(self.shapes[j], stddev=0.01), name=("w%d" % j))
            else:
                self.weights[j] = tf.Variable(tf.zeros(self.shapes[j]), name=("w%d" % j))
        for k in range(len(self.biases) - 1):
            if not zero_model:
#                 self.biases[k] = tf.Variable(tf.zeros([self.neurons]), name=("b%d" % k))
                self.biases[k] = tf.Variable(tf.constant(0.01, shape=[self.neurons]), name=("b%d" % k))
            else:
                self.biases[k] = tf.Variable(tf.zeros([self.neurons]), name=("b%d" % k))
        
        
    def _initTF(self):
        p = self.X
        i = 0
        actf = DNNModel.activations[self.actfunc]
        lossf = DNNModel.losses[self.lossfunc]
        for i in range(0, len(self.weights) - 1):
#             p = tf.nn.relu(tf.matmul(p, self.weights[i]) + self.biases[i])
            p = actf(tf.matmul(p, self.weights[i]) + self.biases[i])
        x = tf.matmul(p, self.weights[i + 1]) + self.biases[i + 1]
        p = tf.nn.sigmoid(x)
        self.predict_op = p
#         self.loss = tf.reduce_mean(tf.square(self.Y - x))
        self.loss = tf.reduce_mean(lossf(x, self.Y))
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        
    def clone(self, other):
        for i in range(len(other.weights)):
            op = self.weights[i].assign(other.session.run(other.weights[i]))
            self.session.run(op)


        
    def distance(self, other):
        dist = 0
        for i in range(len(self.weights)):
            a = self.session.run(self.weights[i])
            b = other.session.run(other.weights[i])
            dist = dist + np.linalg.norm(a - b)
        print "sum distance: " + str(dist)
        return dist / len(self.weights)
    
    
    
    def norm(self):
        n = 0
        for i in range(len(self.weights)):
            n = n + np.linalg.norm(self.session.run(self.weights[i]))
        print "sum norm: " + str(n)
        return n / len(self.weights)
    
    
    def train(self, example):
        x_val, y_val = example
        y_val = np.reshape(y_val, (self.batch_size, 1))
        _, l, p = self.session.run([self.train_op, self.loss, self.predict_op], feed_dict={self.X : x_val, self.Y : y_val})
        return np.average(p), np.average(l) 
        
        
    def getPredictionScore(self, example):
        x_val, _ = example
        return self.session.run(self.predict_op, feed_dict={self.X : x_val})
    
    
    def getLoss(self, example):
        x_val, y_val = example
        return self.session.run(self.train_op, feed_dict={self.X: x_val, self.Y : y_val})
    

    def add(self, other):
        for i in range(len(self.weights)):
            a = self.session.run(self.weights[i])
            b = other.session.run(other.weights[i])
            op = self.weights[i].assign(a + b)
            self.session.run(op)

    
    def scalarMultiply(self, scalar):
        for i in range(len(self.weights)):
            a = self.session.run(self.weights[i])
            op = self.weights[i].assign(a * scalar)
            self.session.run(op)

            
    def getInitParam(self):
        return {"indim" : self.indim, "outdim" : self.outdim, "name" : self.name, "hidden_layers" : self.hidden_layers, "neurons": self.neurons, "batch_size": self.batch_size}
    
    
    def getModelSize(self):
        return self.indim * (self.neurons ** self.hidden_layers) * self.outdim
    
    
    def getModelParameters(self):
        return self.parameters
    
    
    def getModelIdentifier(self):
        return "NNModel"
    
        
    def getZeroModel(self, zero_weights=False):
        kwargs = self.getInitParam()
        if zero_weights:
            kwargs["zero_model"] = True
        zeroModel = self.__class__(**kwargs)
        return zeroModel
    
    
    def fromRecord(self, record):
        return NotImplemented
    
    
    def print_state(self):
        print "model state: begin"
        for i in range(len(self.weights)):
            a = self.session.run(self.weights[i])
            print a
        print "model state: end"


 
    
if __name__ == '__main__':
    v = tf.Variable(tf.zeros([10, 10]))
    

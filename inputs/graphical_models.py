'''
Created on 28.10.2012

@author: Mario Boley
'''

from scipy.stats import bernoulli
from scipy.stats import rv_discrete
from scipy.stats import uniform


from math import log
from inputs import InputStream
from learning import model
import time
from utils.sparse_vector import SparseVector
import random

class ZeroOneConditionedBernoulli(rv_discrete):
    def __init__(self, p0, p1):
        self.zeroBernoulli = bernoulli(p0)
        self.oneBernoulli = bernoulli(p1)
    
    def pmf(self, conditioner, value):
        if conditioner == 0:
            return self.zeroBernoulli.pmf(value)
        else:
            return self.oneBernoulli.pmf(value)
        
    def rvs(self, conditioner):
        if conditioner == 0:
            return self.zeroBernoulli.rvs()
        else:
            return self.oneBernoulli.rvs()

def next_hidden_assignment(assignment):
    i = 0
    while assignment[i] == 1:
        assignment[i] = 0
        i += 1
        if i == len(assignment):
            return False
    assignment[i] = 1
    return True

class BshoutyLongModel(InputStream):
    def __init__(self, dim, drift_prob,num_nodes):
        InputStream.__init__(self, "HiddenLayer" + "(" + str(dim) + ")",num_nodes)
        self.label_distribution = bernoulli(0.5)
        self.dim = dim
        self.drift_prob = drift_prob
        self.hidden_layer_size = int(log(dim, 2))
        self.set_random_parameters()
        self.examples_in_current_macro_round=0
        
    def set_random_parameters(self):
        self.hidden_layer = []
        for _ in xrange(self.hidden_layer_size):
            relevance = uniform.rvs(loc=0.0, scale=1)
            p0 = uniform.rvs(loc=0, scale=1 - relevance)
            p1 = p0 + relevance
            if random.random()<0.5:
                swap=p0
                p0=p1
                p1=swap
            self.hidden_layer.append(ZeroOneConditionedBernoulli(p0, p1))
        self.output_layer = []
        self.effects = []
        for _ in xrange(self.dim):
            relevance = uniform.rvs(loc=0.9, scale=0.1)
            p0 = uniform.rvs(loc=0, scale=1 - relevance)
            p1 = p0 + relevance   
            if random.random()<0.5:
                swap=p0
                p0=p1
                p1=swap         
            self.output_layer.append(ZeroOneConditionedBernoulli(p0, p1))
            self.effects.append(abs(p0 - p1))
            
    def compute_hidden_probs_given_label(self, hidden_assingment, label):
        res = 1
        for i in xrange(len(self.hidden_layer)):
            res *= self.hidden_layer[i].pmf(label, hidden_assingment[i])
        return res

    def computeBayesOptimalError(self):
        res = 0
        assignment = [0] * len(self.hidden_layer)
        new_assignment = True
        while new_assignment == True:
#            print assignment
#            print self.compute_hidden_probs_given_label(assignment, 1)
#            print self.compute_hidden_probs_given_label(assignment, 0)
            res += min(self.compute_hidden_probs_given_label(assignment, 1), self.compute_hidden_probs_given_label(assignment, 0)) / 2.0
            new_assignment = next_hidden_assignment(assignment)
        return res
    
    def get_labels(self):
        return [0, 1]
    
    def get_singletons(self):
        return range(self.dim)
    
    def getIdentifier(self):
        return "HiddenLayer" + "(" + str(self.dim) + ")"
    
    def _generate_example(self):
        self._try_drift()
        label = self.label_distribution.rvs()
        hidden_values = []
        record = set()
        for i in xrange(len(self.hidden_layer)):
            hidden_values.append(self.hidden_layer[i].rvs(label))
            
        for i in xrange(self.dim):
            if self.output_layer[i].rvs(hidden_values[i % len(self.hidden_layer)]) == 1:
                record.add(i)
        if label == 0: label = -1
        record_vector = SparseVector()
        for i in xrange(self.dim):
            if i in record:
                record_vector.components[i] = 1.0
            else:
                record_vector.components[i] = -1.0
        return (record_vector, label)
    
    def _try_drift(self):
        self.examples_in_current_macro_round+=1
        if self.examples_in_current_macro_round==self.number_of_nodes:
            self.examples_in_current_macro_round=0
            if uniform.rvs(loc=0.0, scale=1.0) < self.drift_prob:
                print "DRIFT!!!"
                self.set_random_parameters()
                self._generate_drift_event()

if __name__ == '__main__':
    examples = 10
    start = time.time()
    model = BshoutyLongModel(200, 0.1, 2)
    print model.computeBayesOptimalError()
    for i in xrange(examples):
        print model.generate_example()
        if i+1 % 1000 == 0:
            print ".", 
    end = time.time()
    print "Duration in secs: %d" % int(end - start)
#    condBer=ZeroOneConditionedBernoulli(0.9, 0.1)
#    for i in xrange(10):
#        print condBer.rvs(0)
#    for i in xrange(10):
#        print condBer.rvs(1)

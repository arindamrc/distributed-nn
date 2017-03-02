'''
Created on 07.09.2015

@author: mkamp
'''

import math
from utils import sparse_vector

class SupportVector():
    def __init(self):
        self.record = None
        self.weight = None
        
    def clone(self):
        newSV = SupportVector()
        newSV.weight = self.weight
        newSV.record = self.record.getClone()
        return newSV
    
    def __str__(self):
        return "alpha:"+str(self.weight) + " SVs:"+str(self.record)

class LinearKernel():
    def compute(self, recordA, recordB):
        #Compute the inner product and return the result
        return sparse_vector.dot_product(recordA, recordB)
    
    def getKernelName(self):
        return "Linear"
    
    def getKernelParameters(self):
        return None
    
class PolynomicalKernel():
    def __init__(self, alpha, constant, degree):
        self.alpha = alpha
        self.constant = constant
        self.degree = degree
    
    def compute(self, recordA, recordB):
        return (self.alpha * sparse_vector.dot_product(recordA, recordB) + self.constant) ** self.degree
    
    def getKernelName(self):
        return "Polynomial(d=" + str(self.degree)+")"
    
    def getKernelParameters(self):
        return {'alpha':self.alpha, 'constant':self.constant, 'degree':self.degree}    
    

class GaussianKernel():
    def __init__(self, sigma):
        self.sigma = sigma
        
    def compute(self, recordA, recordB):
        return math.exp(- (recordA.distance_v(recordB) ** 2) / (2 * (self.sigma ** 2)))
        
    def getKernelName(self):
        return "Gaussian(" + str(self.sigma)+")"
    
    def getKernelParameters(self):
        return {'sigma':self.sigma}   
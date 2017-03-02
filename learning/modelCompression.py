'''
Created on 16.02.2016

@author: Michael Kamp
'''

import numpy as np
import copy

class NoCompression():
    def __call__(self, model):
        pass
    
    def getModelCompressionParametersAsText(self):
        return "None"
    
    def getModelCompressionParameters(self):
        return None
       

class TruncationWithThreshold():
    def __init__(self, threshold):
        self.threshold = threshold
    
    def __call__(self, model):
        supportVectors = model.supportVectors
        for vector in supportVectors:
            if(abs(vector.weight) < self.threshold):
                supportVectors.remove(vector)
    
    def getModelCompressionParametersAsText(self):
        return "threshold_" + str(self.threshold)
    
    def getModelCompressionParameters(self):
        return {'threshold':self.threshold}   
          
class TruncationWithSupportVectorsLimitation():
    def __init__(self, limit):
        self.limit = limit
    
    def getModelCompressionParametersAsText(self):
        return "limit_" + str(self.limit)
    
    def __call__(self, model):
        supportVectors = model.supportVectors
        if(len(supportVectors) > self.limit):
            supportVectors.sort(key = lambda x: abs(x.weight), reverse = True)
            del supportVectors[self.limit:]
            
    def getModelCompressionParameters(self):
        return {'limit':self.limit}   

class Projection():
    def __init__(self, epsilon, kernel):
        self.epsilon = epsilon
        self.kernel = kernel
    
    def __call__(self, model):
        supportVectors = model.supportVectors
        if len(supportVectors) <= 2:
            return model
        vt = supportVectors[-1].record
        alphat = supportVectors[-1].weight
        oldSVs = supportVectors[:-1]
        K = np.zeros((len(oldSVs),len(oldSVs)))
        kt = np.zeros(len(oldSVs))               
        for i in xrange(len(oldSVs)):
            vi = oldSVs[i].record
            kt[i] = self.kernel.compute(vi, vt)
            for j in xrange(len(oldSVs)):
                vj = oldSVs[j].record
                K[i][j] = self.kernel.compute(vi,vj)
        Kinv = np.linalg.inv(K)
        dOpt = np.dot(Kinv, kt.T)*alphat
        newModel = copy.deepcopy(model) #only for debug
        newModel.supportVectors = oldSVs
        for i in xrange(dOpt.size):
            newModel.supportVectors[i].weight += dOpt[i]
        dist = newModel.distance(model)        
        if dist < self.epsilon:
            print dist," , ",len(newModel.supportVectors)
            model.supportVectors = newModel.supportVectors
        return model
            
    def getModelCompressionParametersAsText(self):
        return "epsilon(projection)_" + str(self.epsilon)
    
    def getModelCompressionParameters(self):
        return {'epsilon':self.epsilon, 'kernel':self.kernel} 
    
class ProjectionSVLimit():
    def __init__(self, limit, kernel):
        if limit < 2:
            limit = 2
            print "Min. limit is 2. Current limit of ",limit," is thus changed to 2."
        self.limit = limit        
        self.kernel = kernel
    
    def __call__(self, model):
        supportVectors = model.supportVectors
        if len(supportVectors) < self.limit:
            return model
        vt = supportVectors[-1].record
        alphat = supportVectors[-1].weight
        oldSVs = supportVectors[:-1]
        K = np.zeros((len(oldSVs),len(oldSVs)))
        kt = np.zeros(len(oldSVs))               
        for i in xrange(len(oldSVs)):
            vi = oldSVs[i].record
            kt[i] = self.kernel.compute(vi, vt)
            for j in xrange(len(oldSVs)):
                vj = oldSVs[j].record
                K[i][j] = self.kernel.compute(vi,vj)
        Kinv = np.linalg.inv(K)
        dOpt = np.dot(Kinv, kt.T)*alphat
        newModel = copy.deepcopy(model) #only for debug
        newModel.supportVectors = oldSVs
        for i in xrange(dOpt.size):
            newModel.supportVectors[i].weight += dOpt[i]
        dist = newModel.distance(model)        
        print dist," , ",len(newModel.supportVectors)
        model.supportVectors = newModel.supportVectors
        return model
            
    def getModelCompressionParametersAsText(self):
        return "limit(projection)_" + str(self.limit)
    
    def getModelCompressionParameters(self):
        return {'limit':self.limit, 'kernel':self.kernel} 
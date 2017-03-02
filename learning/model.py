'''
Created on 10.01.2013

@author: Michael Kamp
'''

from utils import sparse_vector
from learning.kernel import SupportVector
from copy import deepcopy
# import radonPoints.radonComputation as rc 
import math
import numpy as np

OFFSET_KEY = -1

class Model():
    def __init__(self):
        pass
    
    def clone(self, other):
        pass
        
    def distance(self, other):
        return 0.0
    
    def norm(self):
        return 0.0
    
    def getPredictionScore(self, record):
        return 0.0
    
    def add(self, other):
        pass
    
    def scalarMultiply(self, scalar):
        pass
    
    def getModelSize(self):
        return 0.0
    
    def getModelParameters(self):
        return "None"
    
    def getModelIdentifier(self):
        return "generic_model (please overwrite)"

    def getInitParam(self):
        pass
    
    def predict(self, predictionScore):
        return predictionScore
    
    def getZeroModel(self):
        return self.__class__()
    
    def fromRecord(self, record):
        return self.__class__()
    
class LinearModel(Model):
    def __init__(self, **kwargs):
#        SparseVector.__init__(self)
        self.weight = sparse_vector.SparseVector()
        self.modelType = "linear"
        
    def clone(self, other):
        self.weight.clone(other.weight)
        
    def distance(self, other):
        return self.weight.distance_v(other.weight)
    
    def norm(self):
        return self.weight.norm()
        
    def getPredictionScore(self, record):
        score = sparse_vector.dot_product(self.weight, record)
        score+=self.weight[OFFSET_KEY]
        return score
    
    def add(self, other):
        self.weight.add(other.weight)
    
    def scalarMultiply(self, scalar):
        self.weight.scalar_multiply(scalar)
            
    def getInitParam(self):
        return {}
    
    def getModelSize(self):
        return len(self.weight)
    
    def getModelParameters(self):
        return "None"
    
    def getModelIdentifier(self):
        return "linearModel"
        
    def getZeroModel(self):
        kwargs = self.getInitParam()
        zeroModel = self.__class__(**kwargs)
        return zeroModel
    
    def fromRecord(self, record):
        model = self.getZeroModel()
        model.weight.add(record)
        return model
    
class LinearClassificationModel(LinearModel):
    def predict(self, predictionScore):
        label = 0
        if predictionScore >= 0:
            label = 1
        else:
            label = -1
        return label
    
class LinearRegressionModel(LinearModel):
    def predict(self, predictionScore):
        return predictionScore
    
class DelayedUpdatesLinearModel(LinearModel):
    def __init__(self):
        LinearClassificationModel.__init__(self)
        self.prediction_weights = sparse_vector.SparseVector()
        self.modelType = "stretchedlinear"
    
    def clone(self, other):
        self.weight.clone(other.weight)
        self.prediction_weights.clone(other.weight)
    
    def getPredictionScore(self, record):
        score = sparse_vector.dot_product(self.prediction_weights, record)
        score+=self.weight[OFFSET_KEY]
        return score

class DelayedUpdatesLinearClassificationModel(DelayedUpdatesLinearModel):
    def predict(self, predictionScore):
        label = 0
        if predictionScore >= 0:
            label = 1
        else:
            label = -1
        return label
    
class DelayedUpdatesLinearRegressionModel(DelayedUpdatesLinearModel):
    def predict(self, predictionScore):
        return predictionScore
    
class KernelModel(Model):
    def __init__(self, kernelFunction, modelCompression):
        self.supportVectors = []
        self.kernelFunction = kernelFunction
        self.modelCompression = modelCompression        
        self.modelType = "kernel"
        
    def getPredictionScore(self, record):
        score = 0.0
        for vector in self.supportVectors:
            score += vector.weight * self.kernelFunction.compute(record, vector.record)             
        return score     
    
    def clone(self, other):
        self.supportVectors = deepcopy(other.supportVectors)
        
    def distance(self, other):
        #distance = <w,w> + <r, r> - 2<w, r>
        distance = 0.0            
        for w in self.supportVectors:
            distance += w.weight * w.weight * self.kernelFunction.compute(w.record, w.record)
        for r in other.supportVectors:
            distance += r.weight * r.weight * self.kernelFunction.compute(r.record, r.record)
        for w in self.supportVectors:
            for r in other.supportVectors:
                distance -= 2 * w.weight * r.weight * self.kernelFunction.compute(w.record, r.record)                           
        return distance
    
    def norm(self):
        #sqrt(<w,w>)
        norm = 0.0        
        for w in self.supportVectors:
            norm += w.weight * w.weight * self.kernelFunction.compute(w.record, w.record)
        return math.sqrt(norm)
    
    def scalarMultiply(self, scalar):
        for sv in self.supportVectors:
            sv.weight *= scalar
    
    def add(self, other):
        otherSVs = range(len(other.supportVectors))
        for i in xrange(len(self.supportVectors)):
            sv = self.supportVectors[i]            
            for j in otherSVs:
                otherSv = other.supportVectors[j]
                if otherSv.record == sv.record:
                    self.supportVectors[i].weight += other.supportVectors[j].weight
                    otherSVs.remove(j)
        for j in otherSVs:
            self.supportVectors.append(other.supportVectors[j].clone())
#         vecToIndex = dict()        
#         for i in xrange(len(self.supportVectors)):
#             vecToIndex[self.supportVectors[i].record] = i
#         for i in xrange(len(other.supportVectors)):
#             otherVec = other.supportVectors[i].record
#             if otherVec in vecToIndex.keys():
#                 self.supportVectors[vecToIndex[otherVec]].weight += other.supportVectors[i].weight
#             else:
#                 self.supportVectors.append(other.supportVectors[i].clone())
                
    def getModelSize(self):
        #the weight - one for each support vector - plus the size of each support vector
        messageSize = len(self.supportVectors)
        for vector in self.supportVectors:
            messageSize += vector.record.__len__()
        return messageSize
    
    def getZeroModel(self):
        return self.__class__(**self.getInitParam())
    
    def getInitParam(self):
        return {'kernelFunction':self.getKernelParameters(), 'modelCompression':self.getModelCompressionParameters()}    
    
    def getModelParameters(self):
        return "Kernel Function: " + self.kernelFunction.getKernelName() + ", Truncation Operator: " + self.modelCompression.getModelCompressionParametersAsText()
    
    def getModelIdentifier(self):
        return self.kernelFunction.getKernelName() + "_kernel_trunc_" + self.modelCompression.getModelCompressionParametersAsText()
    
    def getKernelParameters(self):
        if(self.kernelFunction.getKernelParameters() != None):
            return self.kernelFunction.__class__(**self.kernelFunction.getKernelParameters())
        else:
            return self.kernelFunction.__class__()
        
    def getModelCompressionParameters(self):
        if(self.modelCompression.getModelCompressionParameters() != None):
            return self.modelCompression.__class__(**self.modelCompression.getModelCompressionParameters())
        else:
            return self.modelCompression.__class__()

    def fromRecord(self, record):
        model = self.getZeroModel()
        newSupportVector = SupportVector()
        newSupportVector.record = record
        newSupportVector.weight = 1.0
        model.supportVectors.append(newSupportVector)
        return model
     
    def __str__(self):
        return "kernel("+str(self.kernelFunction)+")-"+str(len(self.supportVectors))+" SVs"
     
class KernelRegressionModel(KernelModel):        
    def predict(self, predictionScore):
        return predictionScore


class KernelClassificationModel(KernelModel): 
    def predict(self, predictionScore):
        label = 0
        if predictionScore >= 0:
            label = 1
        else:
            label = -1
        return label
               
def aritmethic_mean(models):
    if len(models) == 0:
        return None
#     models[0].print_state()
    new_model = models[0].getZeroModel(zero_weights = True)
#     models[0].print_state()
    for model in models:
#         model.print_state()
        new_model.add(model)
#     new_model.print_state()
    new_model.scalarMultiply(1./float(len(models)))
#     new_model.scalarMultiply(0.6)
#     new_model.print_state()
    return new_model

# def radon_point(models):
#     r = len(models[0].weight) + 2
#     c = len(models)
#     h = int(math.floor(math.log(c) / math.log(r)))
#     S = []
#     for model in models:
#         S.append(model.weight.toNumpyArray())    
#     S = np.array(S)
#     radonPoint = rc.getRadonPointHierarchical(S, h)
#     new_model = models[0].getZeroModel()
#     new_model.weight.fromNumpyArray(radonPoint)
#     return new_model
    

def dual_average_aritmethic_mean(models):
    if len(models) == 0:
        return None
    new_model = models[0].getZeroModel()
    new_model.last_dual_average = new_model.getZeroModel()
    for model in models:
        new_model.add(model)
        new_model.last_dual_average.add(model.last_dual_average)
    new_model.scalarMultiply(1./float(len(models)))
    new_model.last_dual_average.scalarMultiply(1./float(len(models)))    
    new_model.t = max([model.t for model in models])    
    return new_model
    
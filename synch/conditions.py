'''
Created on 08.09.2015

@author: Michael Kamp
'''

class LocalDivergenceCondition(object):
    def __init__(self,node,referenceModel,threshold):
        self.reference_model=referenceModel
        self.threshold=threshold
        self.node=node
        
    def __call__(self):       
        divergence = self.node.model.distance(self.reference_model)
        return divergence > self.threshold

class LocalDivergenceConditionWithSlack(object):    
    def __init__(self,node,reference_model,threshold):
        self.reference_model=reference_model
        self.threshold=threshold
        self.node=node
        self.slack=0.0
        
    def __call__(self):
        divergence = self.node.model.distance(self.reference_model)
        return divergence + self.slack > self.threshold
    
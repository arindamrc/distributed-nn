'''
Created on Dec 7, 2016

@author: arc
'''

from synch.synchronization import ProbabilisticDriftBasedSync, SynchronizationOperator
from synch.conditions import LocalDivergenceConditionWithSlack
import math
import random as rd

class ProbabilisticDriftBasedSyncNN(ProbabilisticDriftBasedSync):
    def __init__(self, type_string, divergenceThreshold, delta_guarantee):
        self.divergenceThreshold    = divergenceThreshold
        SynchronizationOperator.__init__(self, type_string, str(divergenceThreshold))
        self.deltaGuarantee         = delta_guarantee
        self.activeNodeCounts       = []
        
    def sampleNodes(self, nodes):
        activeNodes = []
        for node in nodes:
            deltaV = math.sqrt(node.model.distance(node.local_condition.reference_model))
            prob = (deltaV / (2.*self.divergenceThreshold*math.sqrt(float(len(nodes)))))*math.sqrt(float(node.model.getModelSize())/self.deltaGuarantee)
            if prob > 1:
                prob = 1            
            val = rd.random()
            if val < prob:
                activeNodes.append((node,prob))
        self.activeNodeCounts.append(len(activeNodes))
        return activeNodes                        
      
    def __call__(self, env, quiet = False):
        nodes_in_violation = []
        self.activeNodes = self.sampleNodes(env.nodes)
        for (node, prob) in self.activeNodes:
            if node.local_condition(1./float(prob)): nodes_in_violation.append(node)      
        if len(nodes_in_violation)==0:
            return        
        self.resolutionMethod(env, nodes_in_violation, quiet)
    
    def resolutionMethod(self, env, affectedNodes, quiet = False):
        pass
    
    def initNodes(self, nodes, env):
        for node in nodes:
            node.ref_model          = env.model.getZeroModel()
            node.local_condition    = LocalDivergenceConditionWithSlack(node, node.ref_model, self.divergenceThreshold)
        self.activeNodes = self.sampleNodes(nodes) 
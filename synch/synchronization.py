from random import shuffle
import numpy as np
import collections as cl
import operator
import math
import time
import random as rd
from loggers.events import Event, generate, COMMUNICATION
from synch.conditions import LocalDivergenceCondition, LocalDivergenceConditionWithSlack
from multiprocessing import Pool

class SynchronizationOperator(object):  
    def __init__(self,type_string,parameter_string):
        self.type   = type_string
        self.parameter_string=parameter_string
    
    def __str__(self):
        if self.parameter_string=="":
            return self.type
        else:
            return self.type+"-"+self.parameter_string

    name = property(__str__, None)
    
    def setModel(self, new_model, nodes):
        for node in nodes:
            node.model.clone(new_model)
    
    def setReferenceModel(self, newReferenceModel, nodes):
        for node in nodes:
            node.local_condition.reference_model.clone(newReferenceModel)
                
    def logCommunication(self, env, nodesThatSentMessages):
        #Increment the message size
        cumulativeMessageSize = 0
        for node in nodesThatSentMessages:
            cumulativeMessageSize += node.model.getModelSize()    
        env.total_message_size  += cumulativeMessageSize
        env.total_communication += len(nodesThatSentMessages)
        
        #Log the message size    
        generate(Event(COMMUNICATION, env.parameters, env.current_round, (cumulativeMessageSize, env.current_round), env.model_identifier)) 
        #env.total_communication+=len(env.nodes)
    
    def initNodes(self, nodes, env):
        pass
            
    def __call__(self,env, quiet = False):
        pass

class CentralSyncOperator(SynchronizationOperator):
    def __init__(self):
        SynchronizationOperator.__init__(self,"central","")
        self.type   = "baseline"
                
class NoSyncOperator(SynchronizationOperator):
    def __init__(self):
        SynchronizationOperator.__init__(self, "nosync", "")
        
class StaticSyncOperator(SynchronizationOperator):    
    def __init__(self):
        SynchronizationOperator.__init__(self,"static","")
        
    def __call__(self, env, quiet = False): 
        print "syncing..."      
        new_model = env.models_aggregation([node.model for node in env.nodes])  
        for node in env.nodes:
            node.model.clone(new_model)
        if not quiet:
            self.logCommunication(env, env.nodes)

class OracleFullSyncOperator(SynchronizationOperator):    
    def __init__(self,divergenceThreshold):
        self.divergenceThreshold    = divergenceThreshold
        self.type                   = "oracle"
        self.parameter_string       = str(divergenceThreshold)

    def __call__(self,env, quiet = False):
        mean_model = env.models_aggregation([node.model for node in env.nodes]) 
        divergence = np.sum([node.model.distance(mean_model) for node in env.nodes])/float(len(env.nodes))
        if divergence > self.divergenceThreshold:                
            self.setModel(mean_model, env.nodes)  
            if not quiet:
                self.logCommunication(env, env.nodes)

# def checkLocalConditionParallel(node):
#     if node.local_condition(): 
#         return node
# 
# pool = None    

class LocalDriftBasedSyncBase(SynchronizationOperator):  
    def __init__(self, type_string, divergence_threshold):
        self.divergenceThreshold    = divergence_threshold
        SynchronizationOperator.__init__(self, type_string, str(divergence_threshold))        
    
    def __call__(self,env, quiet = False):
        nodes_in_violation = []        
#         start = time.time()
#         global pool
#         if pool == None:
#             workers = len(env.nodes)
#             pool = Pool(processes = workers)
#         mylist = pool.map(checkLocalConditionParallel, env.nodes)
#         mylist = [l for l in mylist if l!=None]
#         end = time.time()
#         dur1 = end - start
#         start = time.time()        
        for node in env.nodes:
            if node.local_condition(): nodes_in_violation.append(node)       
        #end = time.time()
        #dur2 = end - start
        #s = set(mylist).intersection(set(nodes_in_violation))
        if len(nodes_in_violation)==0:
            return
        self.resolutionMethod(env, nodes_in_violation, quiet)
    
    def resolutionMethod(self, env, affectedNodes, quiet = False):
        pass
    
    def randomSubset(self, nodes, balancing_set):
        result = []
        nodes_set = set(nodes)    
        balancing_set_set = set(balancing_set)            
        difference = nodes_set.difference(balancing_set_set)
        difference_list = list(difference)
        shuffle(difference_list)
        size = min(len(difference_list), len(balancing_set))
        result += difference_list[0:size]        
        return result
    
    def initNodes(self, nodes, env):
        for node in nodes:
            node.ref_model          = env.modelClass.getZeroModel(zero_weights = True)
            node.local_condition    = LocalDivergenceCondition(node, node.ref_model, self.divergenceThreshold)
    
class ProbabilisticDriftBasedSync(LocalDriftBasedSyncBase):  
    def __init__(self, type_string, divergenceThreshold, delta_guarantee):
        self.divergenceThreshold    = divergenceThreshold
        SynchronizationOperator.__init__(self, type_string, str(divergenceThreshold))
        self.deltaGuarantee         = delta_guarantee
        self.activeNodeCounts       = []
        
    def sampleNodes(self, nodes):
        activeNodes = []
        for node in nodes:
            deltaV = math.sqrt(node.model.distance(node.local_condition.reference_model))
            prob = (deltaV / (2.*self.divergenceThreshold*math.sqrt(float(len(nodes)))))*math.sqrt(float(len(node.model.weight))/self.deltaGuarantee)
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
        
class EagerSynchronization(LocalDriftBasedSyncBase):
    def __init__(self,divergenceThreshold):
        LocalDriftBasedSyncBase.__init__(self, "eager", divergenceThreshold)    

    def resolutionMethod(self,env, affected_nodes, quiet = False):
        newModel = env.models_aggregation([node.model for node in env.nodes])  
        self.setModel(newModel, env.nodes)  
        self.setReferenceModel(newModel, env.nodes)  
        if not quiet:
            self.logCommunication(env, env.nodes)
     
class SlackBalancingSynchronization(LocalDriftBasedSyncBase):    
    def __init__(self,divergenceThreshold):
        LocalDriftBasedSyncBase.__init__(self, "slack", divergenceThreshold)    
    
    def resolutionMethod(self, env, affected_nodes, quiet = False):
        required_slack = np.sum([(node.local_condition.reference_model.distance(node.model) + node.local_condition.slack - node.local_condition.threshold) for node in affected_nodes])
        
        initial_size = len(affected_nodes)
        initial_required_slack = required_slack        
        balancingSet = affected_nodes[:]
        while len(balancingSet) < len(env.nodes) and required_slack > 0.0:
            augmentationSet = self.randomSubset(env.nodes, balancingSet)
            balancingSet += augmentationSet
            required_slack += np.sum([(node.local_condition.reference_model.distance(node.model) + node.local_condition.slack - node.local_condition.threshold) for node in augmentationSet])                           
        final_size = len(balancingSet)        
        if required_slack > 0.0: #            
            self.fullSync(env)
            if not quiet:
                self.logCommunication(env, env.nodes)
        elif len(balancingSet) == len(env.nodes):
            self.sync_reference_model(env.models_aggregation, env.nodes)
            if not quiet:
                self.logCommunication(env, env.nodes)
        else:
            slack_per_node = (required_slack)/float(len(balancingSet))
            for node in balancingSet:
                node.local_condition.slack = self.divergenceThreshold + slack_per_node - node.local_condition.reference_model.distance(node.model)
            if not quiet:
                self.logCommunication(env, balancingSet)
        print "slack resolution: %d\t%d\t%.5f\t%.5f" % (initial_size, final_size, initial_required_slack, self.divergenceThreshold)
        print "slack sum" + str(np.sum([node.local_condition.slack in env.nodes]))

    def fullSync(self, env):
        balanced_model = env.models_aggregation([node.model for node in env.nodes])
        self.setReferenceModel(balanced_model, env.nodes)
        self.setModel(balanced_model, env.nodes)
        self.resetSlack(env.nodes)        
        
    def resetSlack(self,nodes):
        for node in nodes:
            node.local_condition.slack = 0.0
    
    def syncReferenceModel(self,combination, nodes):
        balancedModel = combination([node.model for node in nodes])
        self.setReferenceModel(balancedModel, nodes)
    
class ActiveSynchronization(LocalDriftBasedSyncBase):  
    def __init__(self,divergenceThreshold):
        self.divergenceThreshold    = divergenceThreshold
        self.type                   = "active"
        self.parameter_string       = str(divergenceThreshold)
        
    def resolutionMethod(self,env, affectedNodes, quiet = False):
        referenceModel = env.nodes[0].local_condition.reference_model
        balancingSet = affectedNodes
        balancedModel = env.models_aggregation([node.model for node in balancingSet])
        distance = referenceModel.distance(balancedModel)        
        initial_size = len(affectedNodes)
        initial_distance = distance            
        while len(balancingSet) < len(env.nodes) and \
              distance > self.divergenceThreshold:
            partialNodes = self.randomSubset(env.nodes, balancingSet)
            balancingSet += partialNodes
            balancedModel = env.models_aggregation([node.model for node in balancingSet])
            distance = referenceModel.distance(balancedModel)            
        final_size = len(balancingSet)        
        self.setModel(balancedModel, balancingSet)
        if len(balancingSet) == len(env.nodes): 
            self.setReferenceModel(balancedModel, env.nodes) 
        if not quiet:          
            self.logCommunication(env, balancingSet)    
        print "%s\t%d\t%d\t%.5f" % (self,initial_size, final_size, initial_distance)


class HedgedActiveSync(LocalDriftBasedSyncBase):  
    def __init__(self, divergenceThreshold):
        self.divergenceThreshold    = divergenceThreshold
        self.type                   = "hedge"
        self.parameter_string       = str(divergenceThreshold)
        self.violationCounter      = 0
        
    def resolutionMethod(self, env, affectedNodes, quiet = False):               
        referenceModel = env.nodes[0].local_condition.reference_model
        balancingSet = affectedNodes                             
        balancedModel = env.models_aggregation([node.model for node in balancingSet])      
        distance = referenceModel.distance(balancedModel)           
        initialSize = len(affectedNodes)
        initialDistance = distance
        self.violationCounter+=initialSize        
        if self.violationCounter >= len(env.nodes):
            balancingSet = env.nodes
            self.violationCounter=0
        else:            
            while len(balancingSet) < len(env.nodes) and distance > self.divergenceThreshold:
                balancingSet += self.randomSubset(env.nodes, balancingSet)
                balancedModel = env.models_aggregation([node.model for node in balancingSet])
                distance = referenceModel.distance(balancedModel)                       
        finalSize = len(balancingSet)        
        self.setModel(balancedModel, balancingSet)
        if len(balancingSet) == len(env.nodes): 
            self.setReferenceModel(balancedModel, env.nodes)
        if not quiet:
            self.logCommunication(env, balancingSet)
        print "%s\t%d\t%d\t%.5f" % (self,initialSize, finalSize, initialDistance)

class HedgedDistBaseSync(LocalDriftBasedSyncBase):  
    def __init__(self,divergenceThreshold,fullSyncParam=0.5):
        self.divergenceThreshold=divergenceThreshold
        self.type="hedgeDist"
        self.parameter_string=str(divergenceThreshold)
        self.distances = cl.defaultdict(float)
        self.fullSyncParam = fullSyncParam
        
    def resolutionMethod(self,env, affectedNodes, quiet = False):
        referenceModel = env.nodes[0].local_condition.reference_model
        balancingSet = affectedNodes
        balancedModel = env.models_aggregation([node.model for node in balancingSet])
        distance = referenceModel.distance(balancedModel)
        
        initialSize = len(affectedNodes)
        initialDistance = distance
        for node in affectedNodes:
            self.distances[node] = referenceModel.distance(node.model)
            
        if sorted(self.distances.iteritems(), key=operator.itemgetter(1))[0][1] > self.fullSyncParam*self.divergenceThreshold: #if no node can be found that has a distance smaller than param*threshold than trigger a full sync 
            balancingSet = env.nodes
            self.distances.clear()
        else:
            while len(balancingSet) < len(env.nodes) and distance > self.divergenceThreshold:
                nextNode = sorted(self.distances.iteritems(), key=operator.itemgetter(1))[0][0]                
                self.distances[nextNode] = referenceModel.distance(nextNode.model)
                if self.distances[nextNode] > self.fullSyncParam * self.divergenceThreshold: #again a full sync
                    balancingSet = env.nodes
                    self.distances.clear()
                else:
                    balancingSet += [nextNode]
                balancedModel = env.models_aggregation([node.model for node in balancingSet])
                distance = referenceModel.distance(balancedModel)            
        final_size = len(balancingSet)        
        self.setModel(balancedModel, balancingSet)
        if len(balancingSet) == len(env.nodes): 
            self.setReferenceModel(balancedModel, env.nodes)
        if not quiet:
            self.logCommunication(env, balancingSet)
        print "%s\t%d\t%d\t%.5f" % (self,initialSize, final_size, initialDistance)
        
class ProbabilisticHedgedActiveSync(ProbabilisticDriftBasedSync):  
    def __init__(self,divergenceThreshold, delta_guarantee):
        self.divergenceThreshold    = divergenceThreshold
        self.type                   = "probHedge"
        self.parameter_string       = str(divergenceThreshold)+"-"+str(delta_guarantee)
        self.violationCounter       = 0
        self.deltaGuarantee         = delta_guarantee
        self.activeNodeCounts       = []
        
    def resolutionMethod(self, env, affectedNodes, quiet = False):
        referenceModel = env.nodes[0].local_condition.reference_model
        balancingSet = affectedNodes
        balancedModel = env.models_aggregation([node.model for node in balancingSet])
        distance = referenceModel.distance(balancedModel)        
        initialSize = len(affectedNodes)
        initialDistance = distance
        self.violationCounter += initialSize        
        if self.violationCounter >= len(env.nodes):
            balancingSet = env.nodes
            self.violationCounter=0
        else:
            while len(balancingSet) < len(env.nodes) and distance > self.divergenceThreshold:
                balancingSet += self.randomSubset(env.nodes, balancingSet)
                balancedModel = env.models_aggregation([node.model for node in balancingSet])
                distance = referenceModel.distance(balancedModel)            
        finalSize = len(balancingSet)        
        self.setModel(balancedModel, balancingSet)
        if len(balancingSet) == len(env.nodes): 
            self.setReferenceModel(balancedModel, env.nodes)
        if not quiet:
            self.logCommunication(env, balancingSet)    
        print "%s\t%d\t%d\t%.5f" % (self,initialSize, finalSize, initialDistance)

class ProbabilisticHedgedDistBaseSync(ProbabilisticDriftBasedSync):  
    def __init__(self,divergenceThreshold, delta_guarantee,fullSyncParam=0.5):
        self.divergenceThreshold    = divergenceThreshold
        self.type                   = "probHedgeDist"
        self.parameter_string       = str(divergenceThreshold)+"-"+str(delta_guarantee)
        self.distances              = cl.defaultdict(float)
        self.deltaGuarantee         = delta_guarantee
        self.fullSyncParam          = fullSyncParam
        self.activeNodeCounts       = []
        
    def resolutionMethod(self,env, affectedNodes, quiet = False):
        referenceModel = env.nodes[0].local_condition.reference_model
        balancingSet = affectedNodes
        balancedModel = env.models_aggregation([node.model for node in balancingSet])
        distance = referenceModel.distance(balancedModel)        
        initialSize = len(affectedNodes)
        initialDistance = distance
        for node in affectedNodes:
            self.distances[node] = referenceModel.distance(node.model)            
        if sorted(self.distances.iteritems(), key=operator.itemgetter(1))[0][1] > self.fullSyncParam*self.divergenceThreshold: #if no node can be found that has a distance smaller than param*threshold than trigger a full sync 
            balancingSet = env.nodes
            self.distances.clear()
        else:
            while len(balancingSet) < len(env.nodes) and distance > self.divergenceThreshold:
                nextNode = sorted(self.distances.iteritems(), key=operator.itemgetter(1))[0][0]                
                self.distances[nextNode] = referenceModel.distance(nextNode.model)
                if self.distances[nextNode] > self.fullSyncParam*self.divergenceThreshold: #again a full sync
                    balancingSet = env.nodes
                    self.distances.clear()
                balancingSet += [nextNode]
                balancedModel = env.models_aggregation([node.model for node in balancingSet])
                distance = referenceModel.distance(balancedModel)            
        finalSize = len(balancingSet)        
        self.setModel(balancedModel, balancingSet)
        if len(balancingSet) == len(env.nodes): 
            self.setReferenceModel(balancedModel, env.nodes)
        if not quiet:
            self.logCommunication(env, balancingSet)   
        print "%s\t%d\t%d\t%.5f" % (self,initialSize, finalSize, initialDistance)

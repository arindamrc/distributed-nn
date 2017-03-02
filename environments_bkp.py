from synch.synchronization import NoSyncOperator
from learning.model import aritmethic_mean, LinearModel
# , radon_point
from loggers.events import Event, generate, PREDICTION, UPDATE
from copy import deepcopy
import dnn.model


class PredictionEnvironment():
    def __init__(self, numberOfNodes, updateRule,
                 batchSizeInMacroRounds = None,
                 syncOperator = NoSyncOperator(),
                 model = LinearModel(),                               
                 modelAggregation = aritmethic_mean,
                 lossFunction = None,
                 custom_identifier = None,
                 compute_update_magnitude = False,
                 serial = False):        
        self.node_selector              = NodeSelector()
        self.updateRule                 = updateRule
        self.lossFunction               = lossFunction if lossFunction != None else updateRule.loss_function
        self.models_aggregation         = modelAggregation
        self.batchSizeInMacroRounds     = batchSizeInMacroRounds
        self.syncOperator               = syncOperator
        self.numberOfExamplesProcessed  = 0
        self.current_round              = 1
        self.total_error                = 0.0
        self.total_regression_error     = 0.0
        self.total_communication        = 0
        self.total_message_size         = 0        
        self.modelClass                 = model
        self.model_type                 = model.modelType
        self.model_parameters           = model.getModelParameters()
        self.model_identifier           = model.getModelIdentifier()
        self.custom_identifier          = custom_identifier
        self.computeUpdateMagnitude     = compute_update_magnitude
        self.numberOfNodes              = numberOfNodes
        if serial == True:
            self.nodes                  = self.createNodes(1)
        else:
            self.nodes                  = self.createNodes(numberOfNodes)
        self.serial                     = serial        
        if compute_update_magnitude:
            self.last_hypothesis = model.getZeroModel()
        

    def getIdentifier(self):
        modelAggId = ""
#         if self.models_aggregation != aritmethic_mean:
#             if self.models_aggregation == radon_point:
#                 modelAggId = "_radonPoint"
        if self.custom_identifier is None:
            if len(self.nodes) == 1:
                return "serial_"+str(self.updateRule)+modelAggId
            elif self.syncOperator.type=="nosync":
                return "nosync"
            else:
                base_name=self.syncOperator.name
                name = "%s-%d" % (base_name, self.batchSizeInMacroRounds)
                name += "-"+str(self.updateRule) 
                name += modelAggId                       
                return name
        else:
            return self.custom_identifier
        
    parameters = property(getIdentifier,None)
            
    def createNodes(self, numberOfNodes):
        nodes = []
        for _ in xrange(numberOfNodes):
            node = Node(self)
            node.model = self.modelClass.getZeroModel()            
            nodes.append(node)      
        self.syncOperator.initNodes(nodes, self)      
        return nodes
    
    def process_example(self, example):
        if isinstance(self.modelClass, dnn.model.DNNModel):
            self.process_example_dnn(example)
            return
        record, true_label = example
        node = self.node_selector.get_node(self.nodes)
        predictionScore = node.model.getPredictionScore(record)
        self.total_error += self.lossFunction(true_label, predictionScore)
        self.total_regression_error += abs(true_label-predictionScore)
        generate(Event(PREDICTION, self.parameters, self.current_round, (true_label, predictionScore), node.model.getModelIdentifier()))#, loss,self.current_round)))
        self.updateRule.update(node.model, record, predictionScore, true_label, self.current_round)
        self.numberOfExamplesProcessed += 1
        if self.numberOfExamplesProcessed % self.numberOfNodes == 0:            
            self.synchronize()                        
            self.current_round+=1
            
        if self.computeUpdateMagnitude:
            self.computeUpdateMagnitude()
            
    def process_example_dnn(self, example_batch):
        if self.numberOfExamplesProcessed == 0:
            self._print_model()
        x_val, y_val = example_batch
        y_val = 0 if y_val == -1 else y_val
        x_val = [0 if x == -1 else x for _, x in x_val]
#         x_val = [x for _, x in x_val]
        true_label = y_val
        node, idx = self.node_selector.get_node(self.nodes)
        predictionScore, loss = node.model.train((x_val, y_val))
#         print "predictionScore" + str(predictionScore)
        self.total_regression_error += abs(true_label-predictionScore)
        generate(Event(PREDICTION, self.parameters, self.current_round, (true_label, predictionScore, idx), node.model.getModelIdentifier()))#, loss,self.current_round)))
        self.total_error += loss
        self.numberOfExamplesProcessed += 1
        
        if self.numberOfExamplesProcessed % 500 == 0:
            print predictionScore, loss
            self._print_model()
            
        if self.numberOfExamplesProcessed % self.numberOfNodes == 0:            
            self.synchronize()                     
            self.current_round+=1
            
            
        if self.computeUpdateMagnitude:
            self.computeUpdateMagnitude()
            
    def _print_model(self):
        for node in self.nodes:
            node.model.print_state()
            
    def processExampleForParamEval(self, example):
        record, true_label = example
        node = self.node_selector.get_node(self.nodes)
        predictionScore = node.model.getPredictionScore(record)
        self.total_error += self.lossFunction(true_label, predictionScore)
        self.total_regression_error += abs(true_label-predictionScore)        
        self.updateRule.update(node.model, record, predictionScore, true_label, self.current_round)
        self.numberOfExamplesProcessed += 1
        if self.numberOfExamplesProcessed % len(self.nodes) == 0:            
            self.synchronize(quiet = True)                        
            self.current_round+=1
            
        if self.computeUpdateMagnitude:
            self.computeUpdateMagnitude(quiet = True)
        
    def synchronize(self, quiet = False):
#         print "synchronizing..."
        if self.batchSizeInMacroRounds is None:
            return

        batch_condition_holds = (self.current_round % self.batchSizeInMacroRounds==0)
        if batch_condition_holds:            
            self.syncOperator(self, quiet)
            #print self.nodes[0].model.weights
            
    def computeUpdateMagnitude(self, quiet = False):
        current_hypothesis = self.nodes[0].model
        distance = abs(self.last_hypothesis.distance(current_hypothesis))
        if not quiet:
            generate(Event(UPDATE, self.parameters, self.current_round, (distance, )))
        self.last_hypothesis.clone(current_hypothesis)
        
    def __str__(self):
        name = "PredEnvironment(N:" + str(len(self.nodes)) + " UR:" + str(self.updateRule) + " SO:" + str(self.syncOperator) + ")"
        return name 
        
    def clone(self):
        env = PredictionEnvironment(len(self.nodes), deepcopy(self.updateRule),
                 batchSizeInMacroRounds = self.batchSizeInMacroRounds,
                 syncOperator = deepcopy(self.syncOperator),
                 model = deepcopy(self.modelClass),                               
                 modelAggregation = deepcopy(self.models_aggregation),
                 lossFunction = deepcopy(self.lossFunction),
                 custom_identifier = self.custom_identifier,
                 compute_update_magnitude = self.computeUpdateMagnitude)
        for i in xrange(len(self.nodes)):
            env.nodes[i].clone(self.nodes[i])
        return env
    
class Node():
    index = 0
    def __init__(self,env):
        self.environment=env
        self.model = None
        self.context = None
        self.index = Node.index
        Node.index += 1
        
    def __repr__(self):
        return str(self.index)
    
    def clone(self, other):
        self.model.clone(other.model)
        self.context = deepcopy(other.context)        
        
class NodeSelector():
    def __init__(self):
        self.currentNodeIndex = 0
    
    def get_node(self, nodes):
        result = nodes[self.currentNodeIndex]
        idx = self.currentNodeIndex
        self.currentNodeIndex += 1
        self.currentNodeIndex = self.currentNodeIndex % len(nodes)        
        return result, idx
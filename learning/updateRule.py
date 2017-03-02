from learning.model import OFFSET_KEY
from learning.lossFunction import hinge_loss, epsilon_insensitive_loss, logistic_loss_derivative
from utils import signum
from kernel import LinearKernel, SupportVector
from learning.modelCompression import NoCompression
import math
import itertools


class UpdateRuleBase(object):
    def __init__(self, env):
        self.env = env
    
    def update(self, model, record, prediction_score, true_label, currentRound):
        pass
    
    def getParamRange(self):
        return {}
        
    def getPossParams(self):       
        paramRange = self.getParamRange()
        paramNames = sorted(paramRange)
        possParams = [dict(zip(paramNames, prod)) for prod in itertools.product(*(paramRange[varName] for varName in paramNames))]
        return possParams
    
    def setParams(self, params):
        pass
    
    def __str__(self):
        return "generic update rule (please overwrite)"
    
class StochasticGradientDescent(UpdateRuleBase):
    def __init__(self, lossFunction, lmbda, learning_rate):
        self.loss_function = lossFunction
        self.lmbda = lmbda
        if isinstance(learning_rate, (float, int)):
            self.learning_rate = ConstantLearningRate(learning_rate)
        else:
            self.learning_rate = learning_rate            
        
    def __str__(self):
        return "SGD(loss=%s_lmbd=%s_eta=%s)" % (self.loss_function.shortname, str(self.lmbda), str(self.learning_rate))       
        
    def update(self, model, record, prediction_score, true_label, currentRound):
        if(true_label == model.predict(prediction_score)):
            return
        learningRate = self.learning_rate if not isinstance(self.learning_rate, object) else self.learning_rate(currentRound)
        model.scalarMultiply(self.lmbda)
        der = self.loss_function.derivative(true_label, prediction_score)
        grad = model.fromRecord(record)
        grad.scalarMultiply(der)        
        grad.scalarMultiply(learningRate)
        grad.scalarMultiply(-1.0)
        model.add(grad)
    
    def getParamRange(self):
        possLearningRateVals    = [0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        possLearningRates = []
        for lr in [ConstantLearningRate]:#, DecayingLearningRate]:
            for val in possLearningRateVals:
                possLearningRates.append(lr(val))
        paramRange = {}
        paramRange['lmbda'] = [0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        paramRange['learning_rate'] = possLearningRates
        return paramRange            
    
    def setParams(self, params):
        self.lmbda = params['lmbda']
        self.learning_rate = params['learning_rate']
    
class StretchedStochasticGradientDescent(StochasticGradientDescent):
    def __init__(self, lossFunction, lmbda, learning_rate, stretch_factor):
        self.loss_function = lossFunction
        self.stretch_factor = stretch_factor
        self.lmbda = lmbda
        if isinstance(learning_rate, (float, int)):
            self.learning_rate = ConstantLearningRate(learning_rate)
        else:            
            self.learning_rate = learning_rate            
        
    def __str__(self):
        return "StretchedSGD_sum(loss=%s_lmbd=%s_eta=%s_stretch=%s)" % (self.loss_function.shortname, str(self.lmbda), str(self.learning_rate),str(self.stretch_factor)) 
            
    def update(self, model, record, prediction_score, true_label, currentRound):
        if(true_label == model.predict(prediction_score)):
            return
        model.scalarMultiply(self.lmbda)
        der = self.loss_function.derivative(true_label, prediction_score)
        grad = model.fromRecord(record)
        grad.scalarMultiply(der)
        grad.scalarMultiply(self.learning_rate)
        grad.scalarMultiply(-1.0)
        grad.scalarMultiply(self.stretch_factor)
        model.add(grad)

class DelayedStochasticGradientDescent(StochasticGradientDescent):    
    def __init__(self, base_rule, batch_size = -1): #when batch_size is -1, prediction model is never updated, except for synch, given that the right model class is used
        self.base_rule = base_rule
        self.batch_size = batch_size
        self.examplesInCurrentBatch = {}
        
    def __str__(self):
        return "Delayed" + str(self.base_rule) + "_b" + str(self.batch_size)
        
    def update(self, model, record, prediction_score, true_label, currentRound):
        self.base_rule.update(model, record, prediction_score, true_label)
        if model not in self.examplesInCurrentBatch:
            self.examplesInCurrentBatch[model] = 0
        self.examplesInCurrentBatch[model] += 1
        if self.examplesInCurrentBatch[model] == self.batch_size:
            model.prediction_model.clone(model)
            self.examplesInCurrentBatch[model] = 0

class KernelStochasticGradientDescent(StochasticGradientDescent):
    def __init__(self, lossFunction, lmbda, learningRate):
        self.loss_function = lossFunction
        self.lmbda = lmbda
        if isinstance(learningRate, (float, int)):
            self.learning_rate = ConstantLearningRate(learningRate)
        else:
            self.learning_rate = learningRate  
    
    def update(self, model, record, prediction_score, true_label, currentRound):
        if self.loss_function(true_label, prediction_score) == 0.0:
            return        
        learningRate = self.learning_rate if not isinstance(self.learning_rate, object) else self.learning_rate(currentRound)
        for vector in model.supportVectors:
            vector.weight = (1 - learningRate * self.lmbda) * vector.weight
        weight = - learningRate * self.loss_function.derivative(true_label, prediction_score) 
        newSupportVector = SupportVector()
        newSupportVector.record = record
        newSupportVector.weight = weight
        model.supportVectors.append(newSupportVector)
        
        model.modelCompression(model)

    
    def __str__(self):
        name = "Kernel-SGD(lambda=%s, eta=%s)" % (str(self.lmbda), str(self.learning_rate))        
        return name
        
class PassAgg(UpdateRuleBase):
    def __init__(self, loss_function=hinge_loss, cap_update_rate=None,
                 additive_regularization_param=None):
        self.loss_function = loss_function
        self.cap_update_rate = cap_update_rate
        self.additive_reg_param = additive_regularization_param
 
    def update(self, model, record, prediction_score, true_label, currentRound):
        if(true_label == model.predict(prediction_score)):
            return
        norm = record.norm()
        square_norm = norm * norm
        loss = self.loss_function(true_label, prediction_score)
        update_rate = 0.0
        if self.additive_reg_param != None:
            u = 1 / (2 * self.additive_reg_param)
            d = square_norm + u
            update_rate = loss / d
        else:
            if square_norm > 0:
                update_rate = loss / square_norm
        if self.cap_update_rate:
            update_rate = min(self.cap_update_rate, update_rate)        
        displacement = model.fromRecord(record)
        displacement.scalarMultiply(update_rate)
        displacement.scalarMultiply(true_label)
        model.add(displacement)
    
    def __str__(self):
        baseName = "Passive Aggressive Updater"
        if self.cap_update_rate:
            name = "%s(Update cap: %.4f)" % (baseName, self.cap_update_rate)
        elif self.additive_reg_param:
            name = "%s(Vector regularization: %.4f)" % (baseName, self.additive_reg_param)
        else:
            name = baseName
        
        return name
    
    def getParamRange(self):
        paramRange = {}
        paramRange['cap_update_rate']               = [0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        paramRange['additive_regularization_param'] = [0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        return paramRange            
    
    def setParams(self, params):
        self.cap_update_rate                = params['cap_update_rate']
        self.additive_regularization_param  = params['additive_regularization_param']
        
class PassAggRegression(UpdateRuleBase):
    def __init__(self, loss_function=epsilon_insensitive_loss, epsilon=0.1, cap_regularization_param=None, additive_regularization_param=None):        
        self.loss_function = loss_function
        self.loss_function.epsilon = epsilon
        self.cap_update_rate = cap_regularization_param
        self.additive_reg_param = additive_regularization_param   
        
    def update(self, model, record, prediction_score, true_label, currentRound):
        if(true_label == model.predict(prediction_score)):
            return
        norm = record.norm()
        loss = self.loss_function(true_label, prediction_score)
        if norm > 0.0:
            base_update_rate = loss / (norm * norm)
        else:
            base_update_rate = 0.0
        if self.cap_update_rate is None and self.additive_reg_param is None:
            update_rate = base_update_rate
        elif self.cap_update_rate:
            update_rate = min(self.cap_update_rate, base_update_rate)
        else:
            update_rate = loss / (norm * norm + 1 / 2 * self.additive_reg_param)
        
        displacement = model.fromRecord(record)
        displacement.scalarMultiply(update_rate)
        displacement.scalarMultiply(signum(true_label - prediction_score))
        model.add(displacement)
    
    def __str__(self):
        baseName = "Passive Aggressive Updater - Regression"
        if self.cap_update_rate:
            name = "%s(Update cap: %.4f)" % (baseName, self.cap_update_rate)
        elif self.additive_reg_param:
            name = "%s(Vector regularization: %.4f)" % (baseName, self.additive_reg_param)
        else:
            name = baseName        
        return name

def compute_update_rate_from_data_radius(lmbd, data_radius):
    return 1.0 / (data_radius + lmbd)

class LogisticSGD(UpdateRuleBase):
    def __init__(self, lmbd=1, learning_rate=0.1):
        self.lmbd = lmbd
        self.update_rate = learning_rate
        
    def update(self, model, record, prediction_score, true_label, currentRound):
        if(true_label == model.predict(prediction_score)):
            return
        u = prediction_score
        y = true_label
        lossGrad = logistic_loss_derivative(y, u)
        
        displacement = model.__class__()
        displacement.clone(model)
        displacement.scalarMultiply(self.update_rate)
        displacement.scalarMultiply(self.lmbd)
        displacement.scalarMultiply(-1.0)
        model.add(displacement)

        displacement = model.fromRecord(record)
        displacement.scalarMultiply(self.update_rate)
        displacement.scalarMultiply(lossGrad)
        displacement.scalarMultiply(-1.0)
        model.add(displacement)

    def __str__(self):
        name = "LogisticSGD(lambda=%s, eta=%s)" % (str(self.lmbd), str(self.update_rate))        
        return name
    
    def getParamRange(self):
        possLearningRateVals    = [0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        possLearningRates = []
        for lr in [ConstantLearningRate, DecayingLearningRate]:
            for val in possLearningRateVals:
                possLearningRates.append(lr(val))
        paramRange = {}
        paramRange['lmbda'] = [0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        paramRange['learning_rate'] = possLearningRates
        return paramRange            
    
    def setParams(self, params):
        self.lmbda = params['lmbda']
        self.learning_rate = params['learning_rate']
    
class L2RegularizedUpdate(UpdateRuleBase):
    def __init__(self, updateRule, regParam = 0.1):
        self.updateRule = updateRule
        self.regParam = regParam
    
    def update(self, node, record, prediction_score, true_label, currentRound):
        self.updateRule.update(node, record, prediction_score, true_label, currentRound)
        node.model.scalarMultiply(1.0-self.regParam)

    def __str__(self):
        return "L2Reg_"+self.updateRule.__str__()
    
    def getParamRange(self):
        paramRange = self.updateRule.getParamRange()
        paramRange['regParam'] = [0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        return paramRange            
    
    def setParams(self, params):
        self.updateRule.setParams(params)
        self.regParam = params['regParam']
    
class BallProjectionRegularizedUpdate(UpdateRuleBase):
    def __init__(self, updateRule, regParam = 1.0):
        self.updateRule = updateRule
        self.regParam = regParam 

    def update(self, node, record, prediction_score, true_label, currentRound):
        self.updateRule.update(node, record, prediction_score, true_label, currentRound)        
        if node.model.norm() > 0.0:
            node.model.scalarMultiply(self.regParam/node.model.weight.norm()) #project vector onto ball with radius regParam

    def __str__(self):
        return "L2Reg_"+self.updateRule.__str__()
        
    def getParamRange(self):
        paramRange = self.updateRule.getParamRange()
        paramRange['regParam'] = [0.00001,0.0001,0.001,0.01,0.1,1.,10.0]
        return paramRange            
    
    def setParams(self, params):
        self.updateRule.setParams(params)
        self.regParam = params['regParam']
        
class ConstantLearningRate(object):
    def __init__(self, constant):
        self.constant = constant
        
    def __call__(self, currentRound):
        return self.constant

    def __str__(self):
        return str(self.constant)
    
class DecayingLearningRate(object):
    def __init__(self, factor=0.0):
        self.factor = factor
    
    def __call__(self, currentRound):
        return 1.0 / (self.factor * math.sqrt(currentRound))

    def __str__(self):
        if self.factor == 0:
            return "1(sqrt(t))"
        return "1(" + str(self.factor) + "*sqrt(t))"

import math
epsilon = 0.1

def sign(n):
    if n>=0:
        return 1
    else:
        return -1

def hinge_loss(trueLabel, predictionScore):
    val = trueLabel * predictionScore 
    a = 1.0 - val
    return max(0.0, a)

def hinge_loss_quasi_derivative(trueLabel, predictionScore):
    if trueLabel*predictionScore>1: 
        return 0
    else:
        return -1*trueLabel

def epsilon_insensitive_loss(trueLabel, predictionScore):
    global epsilon    
    value = abs(predictionScore - trueLabel)
    if value <= epsilon:
        return 0.0
    else:
        return value - epsilon
    
def epsilon_insensitive_quasi_derivative(trueLabel, predictionScore):
    global epsilon    
    if abs(predictionScore - trueLabel) <= epsilon:
        return 0.0
    else:
        return  -1*trueLabel + -1*math.copysign(1, trueLabel)*epsilon
    
def logistic_loss(trueLabel, predictionScore):
    try:
        r = math.exp(-1 * trueLabel * predictionScore)
    except OverflowError:
        return -1 * trueLabel * predictionScore
    return math.log(1.0 + r)

def logistic_loss_derivative(trueLabel, predictionScrore):
    try:
        result = -1 * trueLabel / (math.exp(trueLabel * predictionScrore) + 1)
    except OverflowError:
        if trueLabel*predictionScrore<0:
            result=-1*trueLabel
        else:
            result=0
    return result

def squared_loss(trueLabel, predictionScore):
    return 0.5*((trueLabel - predictionScore)**2)

def squared_loss_derivative(trueLabel, predictionScore):
    return predictionScore-trueLabel

class LossFunction(object):
    def __init__(self,function,derivative):
        self.function=function
        self.derivative=derivative
        if not(hasattr(self, "name")): self.name="unknown"
        if not(hasattr(self, "shortname")): self.shortname="unknown"
        
    def __call__(self,trueLabel,predictionScore):
        return self.function(trueLabel,predictionScore)

class LogisticLoss(LossFunction):
    def __init__(self):
        self.name="Logistic Loss"
        self.shortname="logistic"
        LossFunction.__init__(self, logistic_loss, logistic_loss_derivative)      
        
class HingeLoss(LossFunction):
    def __init__(self):
        self.name="Hinge Loss"
        self.shortname="hinge"
        LossFunction.__init__(self, hinge_loss, hinge_loss_quasi_derivative)
        
class SquaredLoss(LossFunction):
    def __init__(self):
        self.name="Squared Regression Loss"
        self.shortname="squared_loss"
        LossFunction.__init__(self, squared_loss, squared_loss_derivative)
        
class EpsilonInsensitiveLoss(LossFunction):
    def __init__(self, eps = 0.1):
        self.name="Epsilon Insensitive Loss"
        self.shortname="eps_insensitive_loss"
        self.eps = eps
        global epsilon    # Needed to modify global copy of globvar
        epsilon = eps
        LossFunction.__init__(self, epsilon_insensitive_loss, epsilon_insensitive_quasi_derivative)
        
class ZeroOneClassificationLoss(LossFunction):
    def __init__(self):
        self.name = "0/1 Error"
        self.shortname = "class_error"
    
    def __call__(self,trueLabel,predictionScore):
        if trueLabel==sign(predictionScore):
            return 0
        else:
            return 1
        
class SmoothedHingeLoss():
    def __init__(self,gamma=1):
        self.gamma=gamma
        self.name="Smoothed Hinge Loss between 1 and "+str(1-gamma)
        self.shortname="smoothed_hinge_"+str(gamma)
        
    def __call__(self,trueLabel,predictionScore):
        if predictionScore*trueLabel>=1-self.gamma:
            return 1/(2*self.gamma)*(max(0,1-predictionScore*trueLabel))**2
        else:
            return 1 - self.gamma/2.0 - predictionScore*trueLabel
        
    def derivative(self,trueLabel,predictionScore):
        if predictionScore*trueLabel >= 1:
            return 0
        if predictionScore*trueLabel < 1-self.gamma:
            return -1*trueLabel
        else:
            return (-1*trueLabel)*(1-predictionScore*trueLabel)*(1.0/self.gamma)
        
        
if __name__ == '__main__':
    print squared_loss(1, 0.50)
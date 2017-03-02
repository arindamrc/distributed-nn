'''
Created on Dec 7, 2016

@author: arc
'''

import sys
sys.path.append("../../../")
import inputs.xor
from inputs.graphical_models import BshoutyLongModel
from learning.lossFunction import HingeLoss, SquaredLoss
from inputs.stoppingConditions import MaxNumberOfExamplesCondition
from learning.updateRule import KernelStochasticGradientDescent,\
    StochasticGradientDescent
from learning.kernel import LinearKernel, GaussianKernel
from learning.modelCompression import NoCompression, Projection
from learning.model import KernelClassificationModel, LinearClassificationModel
from environments import PredictionEnvironment
from synch.synchronization import NoSyncOperator, CentralSyncOperator, StaticSyncOperator, HedgedDistBaseSync
from dnn.model import DNNModel
import experiment
import numpy as np


def main():
    np.random.seed(87655678)

    numberOfNodes   = 5
    dimensionality  = 2
    inputStream     = inputs.xor.XORProblem(nodes = numberOfNodes, dim=dimensionality)
    print "Bayes Optimal Error: ",
    print inputStream.computeBayesOptimalError()
    lossFunction    = SquaredLoss()    
    learningRate    = 0.1
    lmbda           = 0.0001
    sigma           = 4.5
    kernel          = GaussianKernel(sigma)
    noModelComp     = NoCompression()
    modelComp       = Projection(0.1, kernel)
    linupdateRule   = StochasticGradientDescent(lossFunction, lmbda, learningRate)
    updateRule      = KernelStochasticGradientDescent(lossFunction, lmbda, learningRate)
    anchorBatchSize = 20
    linmodel        = LinearClassificationModel()
    model           = KernelClassificationModel(kernel, noModelComp)  
    modelComp       = KernelClassificationModel(kernel, modelComp)  
    dnnModel        = DNNModel(indim = dimensionality, outdim = 1)

    envs    = [
               #baselines
#                 PredictionEnvironment(   numberOfNodes   = numberOfNodes,
#                                          updateRule      = linupdateRule,
#                                          model           = dnnModel,
#                                          batchSizeInMacroRounds  = 5000,
#                                          syncOperator    = NoSyncOperator()),
#                 PredictionEnvironment(   numberOfNodes   = numberOfNodes,
#                                          updateRule      = linupdateRule,
#                                          model           = dnnModel,
#                                          syncOperator    = CentralSyncOperator(),
#                                          serial          = True),
#                 PredictionEnvironment(   numberOfNodes   = numberOfNodes,
#                                          updateRule      = linupdateRule,
#                                          model           = dnnModel,
#                                          syncOperator    = CentralSyncOperator(),
#                                          serial          = False),
#                 #static averaging
#                 PredictionEnvironment(   numberOfNodes           = numberOfNodes,
#                                          updateRule              = linupdateRule,
#                                          model                   = dnnModel,
#                                          batchSizeInMacroRounds  = 5000,#50000 was good
#                                          syncOperator            = StaticSyncOperator()), 
#                 #dynamic averaging    
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linupdateRule,
                                         model                   = dnnModel,
                                         batchSizeInMacroRounds  = 5000,
                                         syncOperator            = HedgedDistBaseSync(1.0),),                          
               ]

    

    #experiment.runParameterEvaluation(inputStream, envs, numberOfNodes*100)
    experiment.run(inputStream, envs, MaxNumberOfExamplesCondition(100))   

     

if __name__ == "__main__":

    main()


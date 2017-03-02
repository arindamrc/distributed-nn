'''
Created on 07.09.2015

@author: Michael Kamp
'''

import sys
sys.path.append("../../../../")

from inputs.finance import DriftStockPrices
from inputs.stoppingConditions import MaxNumberOfExamplesCondition
from learning.lossFunction import EpsilonInsensitiveLoss
from learning.updateRule import StochasticGradientDescent, KernelStochasticGradientDescent
from learning.kernel import LinearKernel, GaussianKernel
from learning.modelCompression import NoCompression, Projection
from learning.model import LinearRegressionModel, KernelRegressionModel
from environments import PredictionEnvironment
from synch.synchronization import NoSyncOperator, CentralSyncOperator, StaticSyncOperator, HedgedDistBaseSync
import experiment

def main():
    #aTargetStocks   = ["ETR:ADS", "ETR:ALV", "ETR:BAS", "ETR:BEI", "ETR:BMW", "ETR:CBK", "ETR:DAI", "ETR:DB1", "ETR:DBK", "ETR:DPW", "ETR:DTE", "ETR:EOAN", "ETR:FME", "ETR:FRE", "ETR:HEI", "ETR:HEN3", "ETR:IFX", "ETR:LHA", "ETR:LIN", "ETR:MEO", "ETR:MRK", "ETR:MUV2", "ETR:RWE", "ETR:SAP", "ETR:SDF", "ETR:SIE", "ETR:TKA", "ETR:VOW3"]
    #inputStream     = StockPriceFeatureStream(dataset = StockPriceFeatureStream.DAX30, label = StockPriceFeatureStream.FV, targetStock=aTargetStocks)    
    #input_stream_pert = InputPerturbatorUnivariateGaussian(input_stream, (0.0,1.0), numberOfNodes)
    #numberOfStocks  = 300
    
    numberOfNodes   = 4      
    anchorBatchSize = 1
    
    inputStream     = DriftStockPrices(nodes = numberOfNodes, driftProb = 0.001, numberOfStocks = None, normalize=True) #driftProb of 0,028 is about once every 8970 examples
    epsilon         = 0.1
    lossFunction    = EpsilonInsensitiveLoss(epsilon)    
    learningRate    = 0.000001
    lmbda           = 0.0001
    sigma           = 1.0
    
    linearUpdateRule= StochasticGradientDescent(lossFunction, lmbda, learningRate)#PassAggRegression()    
    linearModel     = LinearRegressionModel()
    
    epsilon         = 0.001
    
    kernel          = GaussianKernel(sigma)
    truncOp         = Projection(epsilon, kernel)# NoCompression()
    kernelModel           = KernelRegressionModel(kernel, truncOp)     
    kernelUpdateRule      = KernelStochasticGradientDescent(lossFunction, lmbda, learningRate)
    
    envs    = [
               #baselines
                PredictionEnvironment(   numberOfNodes   = numberOfNodes,
                                         updateRule      = linearUpdateRule,
                                         model           = linearModel,
                                         syncOperator    = NoSyncOperator()),
                PredictionEnvironment(   numberOfNodes   = numberOfNodes,
                                         updateRule      = kernelUpdateRule,
                                         model           = kernelModel,
                                         syncOperator    = NoSyncOperator()),
                PredictionEnvironment(   numberOfNodes   = numberOfNodes,
                                         updateRule      = linearUpdateRule,
                                         model           = linearModel,
                                         syncOperator    = CentralSyncOperator(),
                                         serial          = True),
                PredictionEnvironment(   numberOfNodes   = numberOfNodes,
                                         updateRule      = kernelUpdateRule,
                                         model           = kernelModel,
                                         syncOperator    = CentralSyncOperator(),
                                         serial          = True),
                #static averaging
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = 2 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = 4*anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = 8 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()), 
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = 16 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = 32 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),       
               
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = 2 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = 4*anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = 8 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()), 
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = 16 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = 32 * anchorBatchSize,
                                         syncOperator            = StaticSyncOperator()),    
                #dynamic averaging                              
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.05),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.08),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.1),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.3),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.5),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = linearUpdateRule,
                                         model                   = linearModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(1.0),),
               
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.05),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.08),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.1),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.3),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(0.5),),
                PredictionEnvironment(   numberOfNodes           = numberOfNodes,
                                         updateRule              = kernelUpdateRule,
                                         model                   = kernelModel,
                                         batchSizeInMacroRounds  = anchorBatchSize,
                                         syncOperator            = HedgedDistBaseSync(1.0),),
               ]
    
    #experiment.runParameterEvaluation(inputStream, envs, numberOfNodes*100)
    experiment.run(inputStream, envs, MaxNumberOfExamplesCondition(numberOfNodes*500))

if __name__ == "__main__":
    main()
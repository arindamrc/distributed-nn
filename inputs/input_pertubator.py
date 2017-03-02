'''
Created on 12.12.2012

@author: Michael Kamp
'''

from inputs import InputStream
from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
from utils.sparse_vector import SparseVector

class ZeroGenerator(InputStream):
    def __init__(self, dim = 1):
        self.dim = dim
        
    def has_more_examples(self):
        return True
    
    def _generate_example(self):
        record = SparseVector(0.0)
        for i in xrange(self.dim):
            record.components[i] = 0.0
        return (record,0.0)
        
        
class InputPerturbator(InputStream):
    def __init__(self, input_stream):
        self.inputStream = input_stream
    def has_more_examples(self):
        return self.m_oInputStream.has_more_examples()
    def _generate_example(self):
        (record, label) = self.inputStream.generate_example()
        return self.perturb(record, label)
    def perturb(self, record, label):
        return (record, label)
    
    
class InputPerturbatorUnivariateGaussian(InputPerturbator): 
    def __init__(self, input_stream, sigmaRange = (0.0,1.0), macroRoundLength = 1):
        self.inputStream = input_stream
        self.macroLength = macroRoundLength
        self.sigmaRange = sigmaRange
        self.curPos = 0
        self.sigmas = []
        for _ in xrange(self.macroLength):
            map1 = {}
            self.sigmas.append(map1)  
    
    def perturb(self, record, label):
        for key, val in record.components.items():
            if key == -1:
                continue
            if not key in self.sigmas[self.curPos]:
                self.sigmas[self.curPos][key] = uniform.rvs(loc=self.sigmaRange[0], scale=self.sigmaRange[1]-self.sigmaRange[0])
              
            sigma = self.sigmas[self.curPos][key]    
            new_val = val + norm.rvs(loc=0.0, scale=sigma)
            record.components[key] = new_val
        
        self.curPos = (self.curPos + 1) % self.macroLength
        
        return (record, label)
 
 
class InputPerturbatorMultivariateGaussian(InputPerturbator): 
    def __init__(self, input_stream, muRange = (0.0, 0.0), sigmaRange = (0.0,1.0), macroRoundLength = 1):
        self.inputStream = input_stream
        self.macroLength = macroRoundLength
        self.sigmaRange = sigmaRange
        self.muRange = muRange
        self.curPos = 0
        self.covs = []
        self.mus = []
        for _ in xrange(self.macroLength):
            map1 = {}
            self.covs.append(map1)  
            map2 = {}
            self.mus.append(map2) 
    
    def perturb(self, record, label):
        l = len(record.components)
        mu = [0.0]*l 
        sigma = np.array([[0.0]*l]*l)
        for i,key1 in enumerate(record.components):
            if key1 == -1:
                self.mus[self.curPos][key1] = 0.0
            if not key1 in self.mus[self.curPos]:
                    self.mus[self.curPos][key1] = uniform.rvs(loc=self.muRange[0], scale=self.muRange[1]-self.muRange[0])
            mu[i] = self.mus[self.curPos][key1]
            for j,key2 in enumerate(record.components):
                if key2 == -1 or key1 == -1:
                    self.covs[self.curPos][(key1,key2)] = 0.0
                if not (key1,key2) in self.covs[self.curPos]:
                    self.covs[self.curPos][(key1,key2)] = uniform.rvs(loc=self.sigmaRange[0], scale=self.sigmaRange[1]-self.sigmaRange[0])
                sigma[i,j] = self.covs[self.curPos][(key1,key2)]
                
        new_vals = np.array(record.components.values()) + np.random.multivariate_normal(mu, sigma)
        for i,key in enumerate(record.components):
            record.components[key] = new_vals[i]
        
        self.curPos = (self.curPos + 1) % self.macroLength
        
        return (record, label)   
    
    
if __name__ == "__main__":
    #aTargetStocks = ["ETR:ADS"]#, "ETR:ALV", "ETR:BAS", "ETR:BEI", "ETR:BMW", "ETR:CBK", "ETR:DAI", "ETR:DB1", "ETR:DBK", "ETR:DPW", "ETR:DTE", "ETR:EOAN", "ETR:FME", "ETR:FRE", "ETR:HEI", "ETR:HEN3", "ETR:IFX", "ETR:LHA", "ETR:LIN", "ETR:MEO", "ETR:MRK", "ETR:MUV2", "ETR:RWE", "ETR:SAP", "ETR:SDF", "ETR:SIE", "ETR:TKA", "ETR:VOW3"] 
    #inputStream = StockPriceFeatureStream(dataset = StockPriceFeatureStream.DAX30, targetStock=aTargetStocks)
    inputStream = ZeroGenerator(3)
    inputPert = InputPerturbatorMultivariateGaussian(inputStream, (0.0,1.0), (0.0,2.0), 2)
    
    for idx in xrange(20):
        record, label = inputPert.generate_example()
        print record
        print label
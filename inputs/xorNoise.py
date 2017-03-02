'''
Created on 10.03.2016

@author: sbothe
'''

import sys
sys.path.append("../")
import random
from math import sqrt
from utils import sparse_vector
from utils.sparse_vector import SparseVector
from inputs import InputStream

    
class RecordGenerator():
    singletons = []
    
    def __init__(self):    
        pass            
    
    def get_singletons(self):
        return self.singltetons
    
    def generate_record(self):
        return SparseVector()
    
    def getIdentifier(self):
        return ""
    
'''
  Interpret integer as binary vectors, each bit position is a dimension
'''
class HypercubeRecordGenerator(RecordGenerator):
    dim = -1
    
    def __init__(self, dim, noiseEps=0.4 ):
        RecordGenerator.__init__(self)
        self.dim = dim
        self.noiseEps = noiseEps
    
    def generate_record(self):
      retval = SparseVector()
      for i in range(0,self.dim):
        retval.components[i] = random.choice([-1, 1]) + random.uniform(-self.noiseEps, +self.noiseEps)
        
      return retval
        
class SyntheticDataGenerator(InputStream):
    def __init__(self, record_generator, model, nodes = 1):
        self.record_generator = record_generator
        self.model  = model
        InputStream.__init__(self, self.getIdentifier(), nodes)
        
    def getIdentifier(self):
        return self.record_generator.getIdentifier() + "_" + self.model.getIdentifier()
        
    def _generate_example(self):
        record = self.record_generator.generate_record()
        label = self.model.get_label(record)
        return (record, label)
       
class xorModel():
    def __init__(self):
        #nothing todo here
        pass
        
   
    def get_labels(self):
        return set([-1, 1])
    
    def get_label(self, record):
        return self.parity(record)
         

    def parity(self, x):
        ones = 0
        for _, val in x.iteritems():
            #print val
            if (val > 0):
                ones += 1
        if (ones%2 == 1):
            return 1
        else:
            return -1
        
    def getIdentifier(self):
        return 'XOR model'
       
class XORNoiseProblem(SyntheticDataGenerator):
    def __init__(self, examples_per_round=1, nodes = 1, dim=10):
        record_generator = HypercubeRecordGenerator(dim=dim)
        model = xorModel()
        SyntheticDataGenerator.__init__(self, record_generator, model , nodes)
     
    def computeBayesOptimalError( self ):
        return 0.0

def main():
    #input = UniformRandomSum(10, 0.0, 2.0)
    inputStream = XORProblem(dim=2)
    posSeen = 0
    for _ in xrange(100):
        ex, label = inputStream.generate_example()
        
        print (str(ex)+str(' ')+str(label))
        
        if label == 1:
            posSeen += 1
    print('There were '+ str(posSeen)+ ' positive examples')

if __name__ == '__main__':
    main()

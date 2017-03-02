'''
Created on 23.02.2016

@author: Michael Kamp
'''

import unittest
import numpy as np
from learning.kernel import LinearKernel, GaussianKernel, SupportVector
from learning.modelCompression import NoCompression
from learning.model import KernelClassificationModel, aritmethic_mean
from utils.sparse_vector import SparseVector
from utils import sparse_vector

class TestLearners(unittest.TestCase):
    def testKernelModelAveraging(self):
        kernel          = LinearKernel()#GaussianKernel(sigma)
        truncOp         = NoCompression()
          
        
        x1 = SparseVector()
        x1[0] = 1
        x1[1] = 0
        x1[2] = 0
        sv1 = SupportVector()
        sv1.record = x1
        sv1.weight = 0.1
        
        x2 = SparseVector()
        x2[0] = 0
        x2[1] = 1
        x2[2] = 0
        sv2 = SupportVector()
        sv2.record = x2
        sv2.weight = 0.2
        
        x3 = SparseVector()
        x3[0] = 0
        x3[1] = 1
        x3[2] = 0
        sv3 = SupportVector()
        sv3.record = x2
        sv3.weight = 0.3
        
        model1 = KernelClassificationModel(kernel, truncOp)
        model2 = KernelClassificationModel(kernel, truncOp)
        model1.supportVectors.append(sv1)
        model1.supportVectors.append(sv3)
        model2.supportVectors.append(sv2)
        
        model3 = aritmethic_mean([model1, model2])
        self.assert_(model3.supportVectors[0].weight == sv1.weight / 2.0, "Averaging of models failed.")
        self.assert_(model3.supportVectors[1].weight == (sv2.weight + sv3.weight) / 2.0, "Averaging of models failed.")
        
    def testGaussKernel(self):
        sigma = 1.0
        kernel          = GaussianKernel(sigma)
        x1 = SparseVector()
        x1[0] = 1
        x1[1] = 0
        x2 = SparseVector()
        x2[0] = 0
        x2[1] = 1
        
        val1 = kernel.compute(x1, x1)
        self.assert_(val1 == 1.0, "Gaussian kernel of vector with itself is not equal to 1.")
        
        val2 = kernel.compute(x1, x2)
        true_norm = np.linalg.norm(np.array([1,0])-np.array([0,1]))
        true_exp = -1*(true_norm**2)
        true_exp = true_exp / (2*(sigma**2))
        true_val = np.exp(true_exp)
        self.assert_(val2 == true_val, "Gaussian kernel of two vectors is not how I expected it...")
        
    def testLinearKernel(self):
        sigma = 1.0
        kernel = LinearKernel()
        x1 = SparseVector()
        x1[0] = 1
        x1[1] = 0
        x2 = SparseVector()
        x2[0] = 0
        x2[1] = 1
        val = kernel.compute(x1, x2)
        true_val = np.dot(np.array([1,0]), np.array([0,1]))
        self.assert_(val == true_val, "Linear kernel of two vectors is not how I expected it...")
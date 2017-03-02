from inputs import InputStream
from utils import sparse_vector
from random import random

class RapidlyDriftingLinearThreshold(InputStream):
    def __init__(self, dimensionality, radius, margin = 0.0, inside_margin_prob = 1.0, 
                 change_prob = 0.0, examples_per_round=1):
        self.dimensionality = dimensionality
        self.margin = margin
        self.inside_margin_prob = inside_margin_prob
        self.change_prob = change_prob
        self.radius = radius
        self.threshold = self._generate_threshold()
        self.examples_per_round = examples_per_round
        self.examples_this_round = 0
        
    def _generate_threshold(self):
        result = sparse_vector.SparseVector(0.0, add_bias = False)
        for d in xrange(self.dimensionality - 1):
            result.components[d] = random() * 2 * self.radius - self.radius
        result.components[self.dimensionality - 1] = -1
        
        return result
        
    def _generate_example(self):
        example = self._random_weights_example()
        label = self._get_label(example, self.threshold)
        
        result = (example, label)
        
        self.examples_this_round += 1
        if self.examples_this_round >= self.examples_per_round:
            self._try_drift()
            self.examples_this_round = 0
        
        return result
    
    def _random_weights_example(self):
        result = self._random_weights_in_radius_square()
        while not self._in_radius_circle(result):
            result = self._random_weights_in_radius_square()
        return result
    
    def _random_weights_in_radius_square(self):
        result = sparse_vector.SparseVector(0.0, add_bias = False)
        for d in xrange(self.dimensionality):
            result.components[d] = random() * 2 * self.radius - self.radius
        
        return result
    
    def _in_radius_circle(self, model):
        square_radius = self.radius * self.radius
        dSum = 0.0
        for v in model.components.values():
            dSum += v*v            
        return dSum <= square_radius
    
    def _get_label(self, example, threshold):
        u = sparse_vector.dot_product(example, threshold)
        if u >= 0: 
            return 1
        else: 
            return -1
        
    def _try_drift(self):
        if random() <= self.change_prob:
            self.threshold = self._generate_threshold()
            self._generate_drift_event()
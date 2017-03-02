'''
Created on 10.01.2013

@author: Michael Kamp
'''
import math
import numpy as np
from collections import defaultdict

class SparseVector():
    def __init__(self):
        self.components = defaultdict(float)

    def __repr__(self):
        return str(self.components)
    
    def __str__(self):
        return str(self.components.viewitems()).lstrip("dict_items(").rstrip(")")
    
    def __iter__(self):
        return iter(self.components.iteritems())
    
    def __eq__(self, other):
        if len(self.components) == len(other.components):
            for key, value in self.components.iteritems():
                if key not in other.components: return False
                if other.components[key] != value: return False
            return True
        
        for key in set(self.components.keys()).union(other.components.keys()):
            if other.components[key] != self.components[key]:
                return False
            
        return True
    
    def __add__(self, other):
        res = self.createCopy()
        res.add(other)
        return res

    def __sub__(self, other):
        res = self.createCopy()
        res.subtract(other)
        return res
    
    def __iadd__(self, other):
        self.add(other)
        return self
    
    def __isub__(self, other):
        self.subtract(other)
        return self    
    
    def __getitem__(self, index):
        return self.components[index]
    
    def __len__(self):
        return len(self.components)
            
    def norm(self):
        result = 0.0
        for _, v in self.components.iteritems():
            result += v * v
        
        return math.sqrt(result)
    
    def zero_norm(self):
        return len(self.components)
    
    def oneNorm(self):
        result = 0.0
        for _, v in self.components.iteritems():
            result += abs(v)
        return result
    
    def scalar_multiply(self, scalar):
        if scalar == 0:
            self.components.clear()
            return
        for k, v in self.components.iteritems():
            self.components[k] = v * scalar
    
    def add(self, other):
        for k, v in other.components.iteritems():
            self.components[k] += v
    
    def subtract(self, other):        
        for k, v in other.components.iteritems():
            self.components[k] -= v
    
    def distance_v(self, other):
        result = 0.0
        x_components = set(self.components.keys())
        y_components = set(other.components.keys())
        common_components = x_components.intersection(y_components)
        for c in common_components:
            x_val = self.components[c]
            y_val = other.components[c]
            diff = x_val - y_val
            result += diff * diff
            
        for c in x_components.difference(y_components):
            x_val = self.components[c]
            result += x_val * x_val
        
        for c in y_components.difference(x_components):
            y_val = other.components[c]
            result += y_val * y_val
            
        return math.sqrt(result)    

    def getClone(self):
        other = SparseVector()
        other.clone(self)
        return other
                    
    def clone(self, other):
        self.components.clear()
        for k, v in other.components.items():
            self.components[k] = v
            
    def createCopy(self):
        res = SparseVector()
        res.clone(self)
        return res
             
    def __setitem__(self, key, value):
        if value == 0:
            if self.components.has_key(key): self.components.pop(key)
            return
        self.components[key] = value
        
    def iteritems(self):
        return self.components.iteritems()

    def fromNumpyArray(self, x):
        xList = x.tolist()
        return self.fromList(xList)
     
    def getDim(self):
        d = max(self.components.keys())+1
        if -1 in self.components:
            d += 1
        return d
           
    def fromList(self, xList):
        res = SparseVector()
        for i in xrange(len(xList)-1):
            res[i] = xList[i]
        res[-1] = xList[-1] #the convention is that the last element in the list is the intercept. 
        return res
    
    def toList(self):
        return self.toNumpyArray().tolist()        
        
    def toNumpyArray(self):
        d = self.getDim()
        arr = np.zeros(d)
        for key in self.components:
            arr[key] = self.components[key]
        return arr
    
# def random(singletons, radius):
#    pass
            
def dot_product(x, y):
    result = 0.0
    
    x_components = set(x.components.keys())
    y_components = set(y.components.keys())
    common_components = x_components.intersection(y_components)
    for c in common_components:
        xval = x.components[c]
        yval = y.components[c]
        
        result += xval * yval        
    
    return result

def aritmethic_mean_v(vectors):
    if len(vectors) == 0:
        return SparseVector(0.0)
    elif len(vectors) == 1:
        result = SparseVector()
        result.clone(vectors[0])
        return result
    else:
        result = SparseVector()
        all_keys = set()
        for vector in vectors:
            all_keys.update(set(vector.components.keys()))
            
        for c in all_keys:
            value = 0.0
            for vector in vectors:
                value += vector.components.get(c, 0.0)
            value /= len(vectors)
            result.components[c] = value
        
        return result

def average_distance(model, other_models):
    result = 0.0
    for other_model in other_models:
        distance = model.distance(other_model)
        result += distance
        
    return result / len(other_models)

def from_unit_cube(singletons):
    result = SparseVector()
    for singleton in singletons:
        result.components[singleton] = 1.0
        
    return result

def from_record_dictionary(record_dict):
    result = SparseVector()
    for singleton, score in record_dict.iteritems():
        result[singleton] = score
        
    return result

if __name__ == "__main__":
    a = SparseVector()
    # a.components[-1] = 1.0
    a.components["1"] = 1.0
    a.components["2"] = 1.0
    a.components["3"] = 1.0
    
    b = SparseVector()
    # b.components[-1] = 1.0
    b.components["1"] = 4.0
    b.components["2"] = 4.0    
    b.components["3"] = 4.0
    b.components["4"] = 4.0
    
    a.scalar_multiply(0.1)
    b.scalar_multiply(0.2)
    
    print "Dot product: %s" % str(dot_product(a, b))
    print "Squared distance: %s " % str(a.distance_v(b) * a.distance_v(b))
    print "Distance: %s " % str(a.distance_v(b))
    print "Norm(a) = %s" % str(a.norm())
    print "Norm(b) = %s" % str(b.norm())
    print "Dot product(a,a) = %s" % str(dot_product(a, a))
    print "a: %s" % a
    print "b: %s" % b
    print "Average(a,b) = %s" % str(aritmethic_mean_v([a, b]))
    print "Average(a) = %s" % a
    b.clone(a)
    print b
    print "mutiply a by 2"
    b.scalar_multiply(2)
    print b
    
    c = SparseVector()
    c.components["1"] = 3.1
    d = c.createCopy()
    c += a
    print c
    print d
    print c + d

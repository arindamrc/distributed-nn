'''
Created on 10.07.2012

@author: Mario Boley
'''
from random import random, randint, uniform
from math import sqrt
from utils import sparse_vector
from utils.sparse_vector import SparseVector
from inputs import InputStream

def draw_random_subset(S, l):
        res = set([])
        for s in S:
            if random() < l:
                res.add(s)
        return res

class ModelRealValuedSum():
    def __init__(self):
        pass
    
    def getIdentifier(self):
        return "sum"
    
    def get_label(self, record):
        result = 0.0
        for _, v in record.components.iteritems():
            result += v
        return result
    
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
    
class UniformRealValuedRecordGenerator(RecordGenerator):
    def __init__(self, dimensionality, minVal, maxVal):
        RecordGenerator.__init__(self)
        self.singletons = set(range(dimensionality))
        self.minVal = minVal
        self.maxVal = maxVal
    
    def generate_record(self):
        record = SparseVector()
        for key in self.singletons:
            val = uniform(self.minVal, self.maxVal)
            record.__setitem__(key, val)
        return record
    
    def getIdentifier(self):
        return "ur(%d, %s, %s)" % (len(self.singletons), str(self.minVal), str(self.maxVal))

class UniformRecordGenerator(RecordGenerator):
    def __init__(self, number_of_items, inclusion_prob):
        RecordGenerator.__init__(self)
        self.singltetons = set(range(number_of_items))
        self.inclusion_prob = inclusion_prob
    
    def get_average_record_length(self):
        return len(self.singltetons) * self.inclusion_prob
    
    def generate_record(self):
        return draw_random_subset(self.singltetons, self.inclusion_prob)
    
class Disjunction():
    def __init__(self, singletons, literals=set([])):
        self.literals = literals
        self.singletons = singletons
    
    def get_singletons(self):
        return self.singletons
    
    def get_labels(self):
        return set([-1, 1])
    
    def get_label(self, record):
        if not(record.isdisjoint(self.literals)):
            return 1        
        return -1

class ConjunctionOfDisjunctions():
    def __init__(self, singletons, terms):
        self.terms = terms
        self.singletons = singletons
        
    def get_singletons(self):
        return self.singletons
    
    def get_labels(self):
        return set([-1, 1])
    
    def get_label(self, record):
        for term in self.terms:
            if term.get_label(record) == 0:
                return -1
        return 1
    
class LinearThresholdFunction():
    def __init__(self, singletons, weight_max_norm_bound=10):
        self.singletons = singletons
        self.weight_max_norm_bound = weight_max_norm_bound
        self.weight = {}
        self.threshold = 0
        self.set_random_weights()
        
    def set_random_weights(self):
        for s in self.singletons:
            self.weight[s] = random() * self.weight_max_norm_bound
            if random() < 0.5:
                self.weight[s] *= -1
            
    def get_labels(self):
        return set([-1, 1])
    
    def get_label(self, record):
        weight_sum = 0
        for s in record:
            weight_sum += self.weight[s]
        if weight_sum >= self.threshold:
            return 1
        return -1
    
class Drifter():    
    def drift(self, model, stream_identifier):
        if random() < self.change_prob:
            self._do_drift(model)
            return True
        else:
            return False
        
class FixedTermCNFDrifter(Drifter):
    def __init__(self, singletons, num_of_terms, change_prob, disj_inclusion_prob):
        self.singletons = singletons
        self.disj_inclusion_prob = disj_inclusion_prob
        self.change_prob = change_prob
        self.num_of_terms = num_of_terms
        
    def getIdentifier(self):
        return "CNF(%d,%d,%s)" % (len(self.singletons), self.num_of_terms, str(self.change_prob))
    
    def construct_init_model(self):
        terms = []
        for _ in xrange(self.num_of_terms):
            terms.append(Disjunction(self.singletons, draw_random_subset(self.singletons, self.disj_inclusion_prob)))
        return ConjunctionOfDisjunctions(self.singletons, terms)
    
    def _do_drift(self, model):
        model.terms[int(random() * self.num_of_terms)] = Disjunction(self.singletons, draw_random_subset(self.singletons, self.disj_inclusion_prob))
        
    
class RapidDisjunctionDrifter(Drifter):
    def __init__(self, singletons, change_prob, inclusion_prob):
        self.singletons = singletons
        self.change_prob = change_prob
        self.inclusion_prob = inclusion_prob
    
    def getIdentifier(self):
        return "RpDj(%d, %s)" % (len(self.singletons), str(self.change_prob))
    
    def construct_init_model(self):
        return Disjunction(self.singletons, draw_random_subset(self.singletons, self.inclusion_prob))
    
    def _do_drift(self, disjunction):
        disjunction.literals = draw_random_subset(disjunction.get_singletons(), self.inclusion_prob)

class SmoothDisjunctionDrifter(Drifter):
    def __init__(self, singletons, change_prob, inclusion_prob):
        self.singletons = singletons
        self.change_prob = change_prob
        self.inclusion_prob = inclusion_prob

    def getIdentifier(self):
        return "SmDj(%d, %s)" % (len(self.singletons), str(self.change_prob))
    
    def construct_init_model(self):
        return Disjunction(self.singletons, draw_random_subset(self.singletons, self.inclusion_prob))
    
    def _do_drift(self, disjunction):
        disjunction_complement = self.singletons.difference(disjunction.literals)
        l = self.inclusion_prob
        r = l / (1 - l)
        selectOutsideProb = r * len(disjunction_complement) / (len(disjunction.literals) + r * len(disjunction_complement))
        if random() < selectOutsideProb:
            l = list(disjunction_complement)
            changeItem = l[randint(0, len(l) - 1)]
            disjunction.literals.add(changeItem)
#            disjunction_complement.remove(changeItem)
        else:
            l = list(disjunction.literals)
            changeItem = l[randint(0, len(l) - 1)]
#            disjunction_complement.add(changeItem)
            disjunction.literals.remove(changeItem)    

class RapidLinearThresholdDrifter(Drifter):
    def __init__(self, singletons, change_prob):
        self.singletons = singletons
        self.change_prob = change_prob
    
    def getIdentifier(self):
        return "RpLin" + str(self.change_prob)
    
    def construct_init_model(self):
        res = LinearThresholdFunction(self.singletons)
        res.threshold = random()
        if random() < 0.5:
            res.threshold *= -1
        return res
    
    def _do_drift(self, linearThresholdFunction): 
        linearThresholdFunction.set_random_weights()
        linearThresholdFunction.threshold = random()
        if random() < 0.5:
            linearThresholdFunction.threshold *= -1
    
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
    
class UniformRandomSum(SyntheticDataGenerator):
    def __init__(self, dimensionality, minVal, maxVal, examples_per_round=1):
        recordGenerator = UniformRealValuedRecordGenerator(dimensionality, minVal, maxVal)
        model = ModelRealValuedSum()
        SyntheticDataGenerator.__init__(self, recordGenerator, model, examples_per_round)
    
class SyntheticBinaryDataGenerator(SyntheticDataGenerator):
    def __init__(self, record_generator, model_drifter, nodes = 1):
        #self.record_generator = record_generator
        self.model_drifter = model_drifter
        #self.model = model_drifter.construct_init_model()
        self.examples_per_round = nodes
        self.examples_in_current_round = 0
        SyntheticDataGenerator.__init__(self, record_generator, model_drifter.construct_init_model(), nodes)
        
    def _generate_example(self):
        record = self.record_generator.generate_record()
        label = self.model.get_label(record)
        self.examples_in_current_round += 1
        if self.examples_in_current_round == self.examples_per_round:
            has_drifted = self.model_drifter.drift(self.model, self.getIdentifier())
            if has_drifted: self._generate_drift_event()
            self.examples_in_current_round = 0
        return (sparse_vector.from_unit_cube(record), label)
    
    def getIdentifier(self):
        return self.model_drifter.getIdentifier() + "(" + str(len(self.model.singletons)) + ")"


class RapidlyDriftingDisjunction(SyntheticBinaryDataGenerator):
    def __init__(self, number_of_items, change_prob, examples_per_round=1):
        self.item_inclusion_prob = sqrt(1 - pow(0.5, 1 / float(number_of_items)))
        record_generator = UniformRecordGenerator(number_of_items, self.item_inclusion_prob)
        model_drifter = RapidDisjunctionDrifter(record_generator.get_singletons(), change_prob, self.item_inclusion_prob)
        SyntheticBinaryDataGenerator.__init__(self, record_generator, model_drifter, examples_per_round)

class SmoothlyDriftingDisjunction(SyntheticBinaryDataGenerator):
    def __init__(self, number_of_items, change_prob, examples_per_round=1):
        self.item_inclusion_prob = sqrt(1 - pow(0.5, 1 / float(number_of_items)))
        record_generator = UniformRecordGenerator(number_of_items, self.item_inclusion_prob)
        model_drifter = SmoothDisjunctionDrifter(record_generator.get_singletons(), change_prob, self.item_inclusion_prob)
        SyntheticBinaryDataGenerator.__init__(self, record_generator, model_drifter, examples_per_round)
        
class RapidlyDriftingLinearThreshold(SyntheticBinaryDataGenerator):
    def __init__(self, number_of_items, change_prob, examples_per_round=1):
        self.item_inclusion_prob = sqrt(1 - pow(0.5, 1 / float(number_of_items)))
        record_generator = UniformRecordGenerator(number_of_items, self.item_inclusion_prob)
        model_drifter = RapidLinearThresholdDrifter(record_generator.get_singletons(), change_prob)
        SyntheticBinaryDataGenerator.__init__(self, record_generator, model_drifter, examples_per_round)

class DriftingFixedTermCNF(SyntheticBinaryDataGenerator):
    def __init__(self, number_of_items, number_of_terms, changeProb, examples_per_round=1):
        pos_prob_per_term = pow(0.5, 1 / float(number_of_terms))
        item_inclusion_prob = sqrt(1 - pow(1 - pos_prob_per_term, 1 / float(number_of_items)))
        record_generator = UniformRecordGenerator(number_of_items, item_inclusion_prob)
        model_drifter = FixedTermCNFDrifter(range(number_of_items), number_of_terms, changeProb, item_inclusion_prob)
        SyntheticBinaryDataGenerator.__init__(self, record_generator, model_drifter, examples_per_round)

def main():
    #input = UniformRandomSum(10, 0.0, 2.0)
    inputStream = DriftingFixedTermCNF(20, 3, 0.5)
    posSeen = 0
    for _ in xrange(1000):
        ex, label = inputStream.generate_example()
        
        print ex, label
        
        if label == 1:
            posSeen += 1
    print posSeen

if __name__ == '__main__':
    main()

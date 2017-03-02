import re
import pprint
import os
from inputs import InputStream
from models.sparse_vector import SparseVector
from import config
from random import shuffle
import random

COMMENT = '%'
RELATION = '@RELATION'
ATTRIBUTE = '@ATTRIBUTE'
ATTRIBUTE_NOMINAL_TYPE = 'NOMINAL'
ATTRIBUTE_UNKNOWN_TYPE = 'UNKNOWN'
DATA = '@DATA'
UNKNOWN = '?'

ATTRIBUTE_DECODERS = {
                      'NUMERIC': float,
                      'REAL': float,
                      'INTEGER': int,
                      'STRING':str
                      }

class UnknownValue(Exception):
    def __init__(self, attribute_name, raw_value, possible_values):
        message = "Unknown value for attribute %s. Expected values: %s. Encountered value: %s" \
            % (attribute_name, ",".join(possible_values), raw_value)
        Exception.__init__(self, message)

def __decode_attribute_type_info(attribute_type_info):
    if attribute_type_info.upper() in ATTRIBUTE_DECODERS:
        return (attribute_type_info, None)
    else:
        attribute_type = ATTRIBUTE_NOMINAL_TYPE
        attribute_possible_values = attribute_type_info.strip('{} ').split(',')
        return (attribute_type, attribute_possible_values)

def __get_value(attribute_name, attribute_details, raw_value):
    attribute_type, possible_values = attribute_details
    if raw_value == UNKNOWN:
        return raw_value
    
    if attribute_type == ATTRIBUTE_NOMINAL_TYPE:
        if raw_value in possible_values:
            return raw_value
        else:
            raise UnknownValue(attribute_name, raw_value, possible_values)
    else:
        decoder = ATTRIBUTE_DECODERS[attribute_type]
        return decoder(raw_value)


def _decode_data_line(line, arff_description):
    result = {}
    attributes_order = arff_description['attributes_order']
    attributes = arff_description['attributes']
    parts = line.strip().split(',')
    for idx in xrange(len(parts)):
        attribute_name = attributes_order[idx]
        attribute_details = attributes[attribute_name]
        part = parts[idx]
        value = __get_value(attribute_name, attribute_details, part)
        result[attribute_name] = value
        
    return result


def describe(file_name):
    result = {
                   'comments':u'',
                   'relation': u'',
                   'attributes': {},
                   'attributes_order': [],
                   'data_size': 0
                   }
    handle = open(file_name)
    for line in handle:
        line = line.strip()
        
        # Skip blank lines
        if not line: continue
        
        # Comments
        if line.startswith(COMMENT):
            comment = "%s\n" % (re.sub('^\%( )?', '', line))
            result['comments'] += comment
            
        # Relation
        elif line.startswith(RELATION):
            _, relation = re.sub('( |\t)+', ' ', line).split(' ', 1)
            result['relation'] = relation
            
        # Attributes
        elif line.startswith(ATTRIBUTE):
            _, attribute_name, attribute_type_info = re.sub('( |\t)+', ' ', line).split(' ', 2)
            attribute_type, attribute_possible_values = __decode_attribute_type_info(attribute_type_info)
            result['attributes'][attribute_name] = (attribute_type, attribute_possible_values)
            result['attributes_order'].append(attribute_name)
        
        # Skip data
        elif line.startswith(DATA):
            continue
        
        # Count size
        else:
            result['data_size'] += 1
    handle.close()
    
    return result

class ARFF(object):
    def __init__(self, arff_handle, arff_description):
        self.arff_handle = arff_handle
        self.arff_description = arff_description
        self._skip_to_data_section()

    def next(self):          
        try:
            line = self.arff_handle.next()
            return _decode_data_line(line, self.arff_description)
        except StopIteration, ex:
            self.arff_handle.close()
            raise ex
        
    def _skip_to_data_section(self):
        line = self.arff_handle.next()
        while not line.startswith(DATA):
            line = self.arff_handle.next()
        
    def close(self):
        self.arff_handle.close()
        
def open_stream(file_name, arff_description):
    arff_handle = open(file_name)
    return ARFF(arff_handle, arff_description)


def _get_path_to_uci_data_set(file_name):
    data_folder = config.PATH_TO_DATASETS
    uci_data_folder = os.path.join(data_folder, "uci")
    full_dataset_path = os.path.join(uci_data_folder, file_name)
    
    return full_dataset_path

class ARFFInputStream(InputStream):
    def __init__(self, file_name, target_attribute_name, num_nodes=1, positive_target_value=None,
                 nominal_attributes_default_value=1.0, iterations=1):
        InputStream.__init__(self, file_name, num_nodes)
        uci_dataset_path = _get_path_to_uci_data_set(file_name)
#        self.identifier = self._generatate_identifier(file_name)
        self.file_name = file_name
        self.target_attribute_name = target_attribute_name
        self.positive_target_value = positive_target_value
        self.nominal_attributes_default_value = nominal_attributes_default_value
        self._arff_description = describe(uci_dataset_path)
        self.arff_stream = open_stream(uci_dataset_path, self._arff_description)
        self._generated_examples = 0      
        self.iterations = iterations
        self.current_iteration = 1
              
    def _generate_example(self):
        arff_record = self.arff_stream.next()
        self._generated_examples += 1
        example = self._decode_arff_record(arff_record)
        if (self._generated_examples / self.current_iteration == self._arff_description['data_size']) and \
                (self.current_iteration < self.iterations):
            uci_dataset_path = _get_path_to_uci_data_set(self.file_name)
            self.arff_stream = open_stream(uci_dataset_path, self._arff_description)
            self.current_iteration += 1
        return example
        
    def _decode_arff_record(self, arff_record):
        raw_label = arff_record[self.target_attribute_name]
        if self.positive_target_value is None:
            label = raw_label
        else:
            raw_label = str(raw_label)
            if raw_label == str(self.positive_target_value): label = 1
            else: label = -1
        
        result_vector = SparseVector()
        
        for record_attribute_name, record_attribute_value in arff_record.iteritems():
            if record_attribute_name != self.target_attribute_name:
                if record_attribute_value != UNKNOWN:
                    attribute_type = self._get_attribute_type(record_attribute_name)
                    if attribute_type == ATTRIBUTE_NOMINAL_TYPE:
                        vector_feature = "%s-%s" % (record_attribute_name, str(record_attribute_value))
                        result_vector[vector_feature] = self.nominal_attributes_default_value
                    else:
                        result_vector[record_attribute_name] = float(record_attribute_value)
        
        return (result_vector, label)
        
    def has_more_examples(self):
        return self._generated_examples < self._arff_description['data_size'] * self.iterations
    
    def close(self):
        self.arff_stream.close()
    
    def _generatate_identifier(self, file_name):
        return file_name
    
    def _get_attribute_type(self, target_attribute_name):
        attribute_type, _ = self._arff_description['attributes'].get(target_attribute_name, (ATTRIBUTE_UNKNOWN_TYPE, None))
        return attribute_type
    
    def _generate_identifier(self, file_name):
        dot_index = file_name.rfind(".")
        if dot_index == -1:
            return file_name
        else:
            return file_name[0:dot_index]
        
class IIDRandomARFFInputStream(InputStream):
    def __init__(self, file_name, target_attribute_name, num_nodes=1, positive_target_value=None,
                 nominal_attributes_default_value=1.0):
        InputStream.__init__(self, file_name, num_nodes)
        uci_dataset_path = _get_path_to_uci_data_set(file_name)
        self.file_name = file_name
        self.target_attribute_name = target_attribute_name
        self.positive_target_value = positive_target_value
        self.nominal_attributes_default_value = nominal_attributes_default_value
        self._arff_description = describe(uci_dataset_path)
        self.data = self._read_data(uci_dataset_path, self._arff_description)
        self.arff_stream = open_stream(uci_dataset_path, self._arff_description)
        self._number_of_generated_examples = 0
        
    def _read_data(self, uci_dataset_path, arff_description):
        print "Preloading ARFF data..."
        data = []
        
        stream = open_stream(uci_dataset_path, arff_description)
        while True:
            try:
                arff_record = stream.next()
                data.append(self._decode_arff_record(arff_record))
            except StopIteration:
                break
        
        stream.close()
        return data
              
    def _generate_example(self):
        example_index = random.randint(0, len(self.data) - 1)
        example = self.data[example_index]
        self._number_of_generated_examples += 1        
        
        return example
        
    def _decode_arff_record(self, arff_record):
        raw_label = arff_record[self.target_attribute_name]
        if self.positive_target_value is None:
            label = raw_label
        else:
            raw_label = str(raw_label)
            if raw_label == str(self.positive_target_value): label = 1
            else: label = -1
        
        result_vector = SparseVector()
        
        for record_attribute_name, record_attribute_value in arff_record.iteritems():
            if record_attribute_name != self.target_attribute_name:
                if record_attribute_value != UNKNOWN:
                    attribute_type = self._get_attribute_type(record_attribute_name)
                    if attribute_type == ATTRIBUTE_NOMINAL_TYPE:
                        vector_feature = "%s-%s" % (record_attribute_name, str(record_attribute_value))
                        result_vector[vector_feature] = self.nominal_attributes_default_value
                    else:
                        result_vector[record_attribute_name] = float(record_attribute_value)
        
        return (result_vector, label)
        
    def has_more_examples(self):
        return True
    
    def _get_attribute_type(self, target_attribute_name):
        attribute_type, _ = self._arff_description['attributes'].get(target_attribute_name, (ATTRIBUTE_UNKNOWN_TYPE, None))
        return attribute_type
    
    def _generate_identifier(self, file_name):
        dot_index = file_name.rfind(".")
        if dot_index == -1:
            return file_name
        else:
            return file_name[0:dot_index]
        
if __name__ == "__main__":
    file_name = '../../data/uci/gas-sensor-array.arff'
    description = describe(file_name)
    pprint.pprint(description)
    #stream = open_stream(file_name, description)
    #for x in xrange(description['data_size']):
    #    record = stream.next()
    #    if x % 100000 == 0:
    #        pprint.pprint(record)
    #        print x / 100000
    #stream.close()
    print "Finished ARFF test"
    
    #dataset = ARFFInputStream('gas-sensor-array.arff', 'class', positive_target_value = 2)
    #i = 0
    #l = 0
    #n = 0
    #while dataset.has_more_examples():
    #    record, label = dataset.generate_example()
    #    if label == 1: l += 1
    #    else: n += 1
    #    i += 1
        
    #    if i % 10000 == 0:
    #        print record
        
    #print "Total number of examples: %d" % (i)
    #print "Positive: %d" % l
    #print "Negative: %d" % n
    
    dataset = IIDRandomARFFInputStream('shuttle.arff', 'class', positive_target_value = 1)
    i = 0
    l = 0
    n = 0
    for i in xrange(1000000):
        record, label = dataset.generate_example()
        if label == 1: l += 1
        else: n += 1
        i += 1
        
        if i % 10000 == 0:
            print record
        
    print "Total number of examples: %d" % (i)
    print "Positive: %d" % l
    print "Negative: %d" % n    

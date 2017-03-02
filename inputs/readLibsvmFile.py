import os
import random
from inputs import InputStream
from utils import sparse_vector
import config

class LibsvmStream(InputStream):
    def __init__(self, data_file='higgs/higgs.svm', nodes = 1, repetitions = 1):
        InputStream.__init__(self,"libsvm_reader(data_file = %s)" % data_file, nodes)
        self.file_name = os.path.join(config.PATH_TO_DATASETS, data_file)
        self.labels = set([1,-1])
        self.input_file_handle = open(self.file_name, 'r')
        self.current_line = self._fetch_line()

    def has_more_examples(self):
        return self.current_line is not None

    def _fetch_line(self):
        try:
            return self.input_file_handle.next()
        except StopIteration:
            return None


    def _generate_example(self):
        record, label = self._decode_line(self.current_line)
        self.current_line = self._fetch_line()

        return (record, label)

    def _decode_line(self, line):
        tokens = line.split()

        # first position is label
        label = float(tokens.pop(0))
        label = (label - 0.5) * 2
        # sparse features

        record = sparse_vector.SparseVector()
        for i in range(len(tokens)):
          (index, value) =  tokens[i].split(':')
          record[index] = float(value)


        return (record, label)

    def close(self):
       self.input_file_handle.close()

    def __deepcopy__(self, memo):
        self.input_file_handle.seek(0)
        return self

if __name__=="__main__":
    stream = LibsvmStream(data_file = 'higgs.svm', nodes = 1, repetitions=1)

    count = 0
    for idx in range(100):
        count += 1
        example = stream.generate_example()
        if count % 10 == 0:
            print count
            print example
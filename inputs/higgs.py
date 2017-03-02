'''
Created on Feb 16, 2017

@author: arc
'''

import numpy as np
import tensorflow as tf
import math as math
import argparse
from  inputs import InputStream
from utils.sparse_vector import SparseVector


class HiggsStream(InputStream):
    def __init__(self, source, identifier, nodes=1, batch_size=100):
        self.generated_examples = 0
        self.current_round = 1
        self.number_of_nodes = nodes
        self._identifier = identifier
        self.batch_size = batch_size
        self.source = source
        self._generated_examples = 0
        self._setup()
        
    def _setup(self):
        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.examples, self.labels = self._input_pipeline(self.batch_size, None)
        # start populating filename queue
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)
        
    def _read_from_csv(self, filename_queue):
        reader = tf.TextLineReader(skip_header_lines=1)
        _, csv_row = reader.read(filename_queue)
#         record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        record_defaults = [[0.0]] * 29
        print record_defaults
        record = tf.decode_csv(csv_row, record_defaults=record_defaults)
#         features = tf.pack(record[1:])  
        features = tf.stack(record[1:])
#         label = tf.pack(record[0])
        label = tf.stack(record[0])
        return features, label
    
    def _input_pipeline(self, batch_size, num_epochs=None):
        filename_queue = tf.train.string_input_producer([self.source], num_epochs=num_epochs, shuffle=True)  
        example, label = self._read_from_csv(filename_queue)
        min_after_dequeue = self.batch_size * 10
        capacity = min_after_dequeue + 3 * self.batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=self.batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return example_batch, label_batch
         

    def _get_sparse_rep(self, example_batch):
        sv_list = []
        for ex in example_batch:
            rec = SparseVector()
            for i in range(len(ex)):
                rec[i] = ex[i]
            sv_list.append(rec)
        return sv_list
    
    
    def _generate_example(self):
        self._generated_examples += self.batch_size
        example_batch = None
        label_batch = None
        sv_example_batch = None
        try:
            if not self.coord.should_stop():
                example_batch, label_batch = self.sess.run([self.examples, self.labels])
                sv_example_batch = self._get_sparse_rep(example_batch)
        except tf.errors.OutOfRangeError:
            print('Done training, epoch reached')
        return example_batch, label_batch
        
    def finalize(self):
        try: 
            self.coord.request_stop()
            self.coord.join(self.threads) 
        finally:
            self.sess.close()
        
if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('dataset')
#     args = parser.parse_args()
#     print(args.dataset)
    st = HiggsStream("/home/arc/Desktop/shared/Uni Bonn Study Materials/Data Science Lab/HIGGS.csv", "higgs", batch_size=10)
    for i in range(10):
        ex, l = st.generate_example()
        print(ex)
        print len(ex[0])
        print(l)
    st.finalize()

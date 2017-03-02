from distributedOnlineLearning.inputs import InputStream
from distributedOnlineLearning.utils import sparse_vector
from distributedOnlineLearning import config
import os

class FinanceNews(InputStream):
    def __init__(self, data_file, frequency = True, normalize = True, nodes = 1):
        InputStream.__init__(self,"FinanceNews(data_file = %s)" % data_file, nodes)
        self.frequency = frequency
        self.normalize = normalize
        self.file_name = os.path.join(config.PATH_TO_FINANCIAL_NEWS, data_file)
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
        parts = line.strip().split("\t")
        record = {}
        label = int(parts[1])
        for part in parts[3:]:
            feature, score = part.split(",")
            try:
                feature_code = int(feature)
            except ValueError:
                feature_code = str(feature)
            
            if self.frequency:
                score = float(score)
            else:
                score = 1.0
                
            record[feature_code] = score
        
        if self.normalize:
            result = sparse_vector.from_record_dictionary(self._normalize(record))
        else:
            result = sparse_vector.from_record_dictionary(record)
            
        return result, label
    
    def _normalize(self, record):
        factor = sum(record.values())
        result = {}
        for k, v in record.iteritems():
            result[k] = float(v) / factor
            
        return result
    
    def close(self):
        self.input_file_handle.close()      
        
if __name__ == "__main__":
    stream = FinanceNews(data_file = 'reuters_eurusd_minute_10000.csv')
    count = 0
    while stream.has_more_examples():
        ex, label = stream.generate_example()
        if count % 1000 == 0:
            print label
            print ex
        count += 1
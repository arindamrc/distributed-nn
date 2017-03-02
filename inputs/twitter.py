import simplejson as json
import os
import random
from distributedOnlineLearning.inputs import InputStream
from distributedOnlineLearning.utils import sparse_vector
from distributedOnlineLearning import config

class TwitterStream(InputStream):
    def __init__(self, data_file='1000_tweets_24_hours', nodes = 1, repetitions = 1):
        InputStream.__init__(self,"Twitter(data_file = %s)" % data_file, nodes)
        self.file_name = os.path.join(config.PATH_TO_TWITTER, data_file)
        self.labels = set([1,-1])
        self.input_file_handle = open(self.file_name, 'r')
#        self.identifier = 
        self.current_line = self._fetch_line()
        self.current_episode = None
        self.repetitions = repetitions
        self.current_repetition = 0
    
    def has_more_examples(self):
        return self.current_line is not None
    
    def _fetch_line(self):
        try:
            return self.input_file_handle.next()
        except StopIteration:
            self.current_repetition += 1
            if self.current_repetition >= self.repetitions:
                return None
            else:
                self.input_file_handle.close()
                self.input_file_handle = open(self.file_name, 'r')
                return self.input_file_handle.next()
    
    def _generate_example(self):
        record, label = self._decode_line(self.current_line)
        self.current_line = self._fetch_line()
        
        return (record, label)
    
    def _decode_line(self, line):
        tweet_dict = json.loads(line)
        record_singletons = self.generate_features(tweet_dict["features"])
        record = sparse_vector.from_unit_cube(record_singletons)
        label = tweet_dict["label"]
        episode = tweet_dict.get("episode", None)
        self._test_episode_drift(episode)
        
        return (record, label)
        
    def _test_episode_drift(self, episode):
        if episode:
            if self.current_episode is None:
                self.current_episode = episode
            elif self.current_episode != episode:
                self.current_episode = episode
                self._generate_drift_event()
    
    def generate_features(self, features):
        result = set()
        for feature in features:
            if isinstance(feature, list):
                result.add(tuple(feature))
            else:
                result.add(feature)
        return result
    
    def close(self):
        self.input_file_handle.close()
        
class RandomIIDTwitterStream(TwitterStream):
    def __init__(self, data_file='1000_tweets_24_hours', nodes = 1):
        InputStream.__init__(self,"RandomIIDTwitter(data_file = %s)" % data_file, nodes)
        self.file_name = os.path.join(config.PATH_TO_TWITTER, data_file)
        self.labels = set([1,-1])
        self.data = self._read_data(self.file_name)
        
    def _read_data(self, file_name):
        print "Preloading data..."
        result = []
        f = open(self.file_name, 'r')
        for line in f:
            example = self._decode_line(line)
            result.append(example)
            
        print "Preloading data complete. Size: %d" % len(result)
        return result
    
    def _generate_example(self):
        index = random.randint(0, len(self.data) - 1)
        return self.data[index]
    
    def has_more_examples(self):
        return True
    
    def close(self):
        pass
        
        
if __name__=="__main__":
    stream = TwitterStream(data_file = 'median_min_9_per_author_1000_features', nodes = 1, repetitions=2)
    
    stream = RandomIIDTwitterStream(data_file = 'median_min_9_per_author_1000_features', nodes = 1)
    
    count = 0
    for idx in range(100000):
        count += 1
        example = stream.generate_example()
        if count % 1000 == 0:
            print count
            print example
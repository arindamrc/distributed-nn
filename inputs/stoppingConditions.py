
class MaxNumberOfExamplesCondition():
    def __init__(self, number_of_examples):
        self.number_of_examples = number_of_examples
    
    def __call__(self, stream):
        return not (stream.generated_examples <= self.number_of_examples and stream.has_more_examples())
#        current_round = time_control.current_example
#        return not (current_round <= self.number_of_examples and stream.has_more_examples())
    
class EndOfStreamCondition():
    def __call__(self, stream):
        return not stream.has_more_examples()
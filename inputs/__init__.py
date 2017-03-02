import loggers.events as events
import random
from loggers.events import Event
                
class InputStream(object):
    
    def __init__(self,identifier,nodes=1):
        self.generated_examples=0
        self.current_round=1
        self.number_of_nodes=nodes
        self._identifier=identifier
         
    def getIdentifier(self):
        return self._identifier

    def set_identifier(self,identifier):
        self._identifier=identifier
        
    parameters=property(getIdentifier,set_identifier,None,"parameters of InputStream that is, e.g., used in experiment logs")

    def generateExampleForParamEval(self):
        return self._generate_example()        
    
    def generate_example(self):
        self.generated_examples+=1
        if self.generated_examples % self.number_of_nodes==0:
            if (self.current_round) % 1000 == 0:
                print "Completed round: "+str(self.current_round)
            self.current_round+=1
        return self._generate_example()
    
    def has_more_examples(self):
        return True
    
    def _generate_drift_event(self):
        events.generate(Event("drift", 'global', self.current_round, (), ""))
        
    def reset(self):
        self.current_round = 0
        self.generated_examples = 0
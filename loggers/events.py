event_queue = []
#event_transformations = []

PREDICTION = "prediction"
COMMUNICATION = "communication"
UPDATE = "update"

def clear_event_queue():
    global event_queue
    event_queue = []

def generate(event):    
    global event_queue
    event_queue.append(event)
    
        
class Event():
    def __init__(self, event_type, locality, time, arguments, modelType):
        self.event_type = event_type
        self.locality = locality
        self.arguments = arguments
        self.time = time
        self.modelType = modelType
        

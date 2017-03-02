import os
import time
from output import outputGenerator
from loggers.events import event_queue

handles = {}

def get_file_handle(event):
    global handles
    event_file = "%s_%s_%s.log" % (event.event_type, event.locality, event.modelType)
    file_name = os.path.join(outputGenerator.current_logs_folder, event_file)    
    if file_name in handles:
        return handles[file_name]
    else:
        handle = open(file_name, 'w')
        handles[file_name] = handle
        return handle
    
def close_handles():
    global handles
    for handle in handles.values():
        handle.close()

def log_event(event):
    h = get_file_handle(event)
    event_arguments = "\t".join([str(event.time)] + [str(arg) for arg in list(event.arguments)])
    h.write("%s\n" % event_arguments)
    
last_report_at = time.time()

def process_round():
    #global event_queue
    while len(event_queue) > 0:
        event = event_queue.pop()
        log_event(event)
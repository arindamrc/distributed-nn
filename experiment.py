import time
from output import outputGenerator
from loggers import logs
from synch.synchronization import HedgedActiveSync, NoSyncOperator
import copy 

time_format = '%Y-%m-%d %H-%M-%S'
current_timestamp = ""

def outputStartTime():
    start_time = time.time()
    timestamp = time.strftime(time_format, time.localtime(start_time))
    print "Experiment started at %s" % (timestamp)
    
    return start_time

def outputEndTime(start_time):
    end_time = time.time()
    print "Experiment ended at %s" % (time.strftime(time_format, time.localtime(end_time)))
    seconds = end_time - start_time
    minutes = int(seconds) / 60
    
    rest = int(seconds - minutes * 60)
        
    print "Duration: %d minutes and %d seconds" % ((int(minutes)), int(rest))
    
def setCurrentTimestamp(start_time):
    global current_timestamp
    current_timestamp = time.strftime(time_format, time.localtime(start_time))

def run(input_stream, envs, stopping_condition, report_at = 10000):
    global current_round, current_timestamp
    start_time = outputStartTime()
    setCurrentTimestamp(start_time)    
    outputGenerator.create_experiment_folder_structure(current_timestamp)
    outputGenerator.experiment_summary(current_timestamp, input_stream, envs)
    
    while(not stopping_condition(input_stream)):
        example = input_stream.generate_example()
        for env in envs:
            env.process_example(example)
        logs.process_round()

    outputEndTime(start_time)
    outputGenerator.experiment_number_of_rounds(current_timestamp,input_stream.current_round-1)
    logs.close_handles()
    outputGenerator.write_experiment_result_file(current_timestamp, envs)
    outputGenerator.output_generate_chart_script(current_timestamp, envs)

def runParameterEvaluation(origInputStream, envs, numberOfExamples):
    inputStream = copy.deepcopy(origInputStream)
    print "Start parameter evaluation:"
    for env in envs:
        paramEvalEnvironment = env.clone()
        print paramEvalEnvironment, " ",        
        updateRule = paramEvalEnvironment.updateRule
        possParams = updateRule.getPossParams()
        errors = {}
        for idx in xrange(len(possParams)):
            exampleCount = 0
            print ".",
            params = possParams[idx]
            paramEvalEnvironment.updateRule.setParams(params)
            while (exampleCount <= numberOfExamples):
                example = inputStream.generate_example()
                paramEvalEnvironment.processExampleForParamEval(example)                
                exampleCount += 1
            errors[idx] = paramEvalEnvironment.total_error
            paramEvalEnvironment.total_error = 0.0
            inputStream.reset()    
        bestParamIdx = min(errors, key=errors.get)
        bestParam = possParams[bestParamIdx]
        paramStr = "{"
        for param in bestParam:
            paramStr += str(param) + ":" + str(bestParam[param]) + ", "
        paramStr = paramStr[:-2] + "}"
        print ""
        print "     best params: ", paramStr
        env.updateRule.setParams(bestParam)
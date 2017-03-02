import utils as helpers
import os
from parseOutput import NUMBER_OF_ROUNDS_KEY

current_folder = ""
current_logs_folder = ""
current_charts_folder = ""

CHART_GENERATION_SCRIPT_TEXT = r"""
from output.plotting import plot_error, plot_communication, plot_performance, plot_accuracy
from learning.lossFunction import SmoothedHingeLoss, ZeroOneClassificationLoss, SquaredLoss, LogisticLoss
from output.parseOutput import ENV_REGRESSION_ERROR_COLUMN, ENV_ERROR_COLUMN
import os

if __name__ == "__main__":
    print os.getcwd()
    plot_performance.generate(os.getcwd()+"/",ENV_ERROR_COLUMN)
    plot_error.generate(os.getcwd()+"/",True,True,ZeroOneClassificationLoss())
    plot_accuracy.generate(os.getcwd()+"/",True,True,ZeroOneClassificationLoss())    
    plot_communication.generate(os.getcwd()+"/",True,True)
"""

def create_experiment_folder_structure(experiment_timestamp):
    timestamp_folder_path = helpers.create_timestamp_folder(experiment_timestamp)
    logs_folder_path = os.path.join(timestamp_folder_path, "logs")
    os.mkdir(logs_folder_path)
    charts_folder_path = os.path.join(timestamp_folder_path, "charts")
    os.mkdir(charts_folder_path)
    
    global current_folder, current_logs_folder, current_charts_folder
    current_folder = timestamp_folder_path
    current_logs_folder = logs_folder_path
    current_charts_folder = charts_folder_path

def experiment_input_source(summary_handle, input_stream):
    summary_handle.write("Input stream: %s\n\n" % input_stream.getIdentifier())

def experiment_environment(file_handle, env):
    file_handle.write("%s\n" % env.parameters)
    file_handle.write("\tModel Type:%s\n" % env.model_type)
    file_handle.write("\tModel Parameters:%s\n" % env.model_parameters)
    file_handle.write("\tNumber of nodes: %d\n" % len(env.nodes))
    file_handle.write("\tUpdate rule: %s\n" % env.updateRule)
    if env.batchSizeInMacroRounds:
        file_handle.write("\tBatch size (in macro rounds): %d\n" % env.batchSizeInMacroRounds)
    file_handle.write("\tSync. operator: %s\n" % env.syncOperator)

def experiment_environments(file_handle, envs):
    file_handle.write("Environments: \n")
    for env in envs:
        experiment_environment(file_handle, env)

RESULT_TITLE="#Identifier\t"+"Label\t"+"Type\t"+"Parameters\t"+"Error\t"+"Regression_Error\t"+"Communication\t"+"Message Size\t"+"\n"

def experiment_summary(experiment_timestamp, input_stream, envs):
    timestamp_folder_path = helpers.get_timestamp_folder(experiment_timestamp)
    
    summary_file_path = os.path.join(timestamp_folder_path, "summary.txt")
    summary_handle = open(summary_file_path, 'w')
    
    experiment_input_source(summary_handle, input_stream)
    experiment_environments(summary_handle, envs)
    
    summary_handle.close()
    
    envs_file_path = os.path.join(timestamp_folder_path, "list_of_envs.txt")
    handle = open(envs_file_path, 'w')
    for env in envs:
        handle.write("%s\n" % env.parameters)
    handle.close()

def write_experiment_result_file(experiment_timestamp, envs):
    timestamp_folder_path = helpers.get_timestamp_folder(experiment_timestamp)

    result_file_path = os.path.join(timestamp_folder_path, "results.txt")
    handle = open(result_file_path, 'w')
    handle.write(RESULT_TITLE)
    for env in envs:
        handle.write("%s\t%s\t%s\t%s\t%f\t%f\t%i\t%i\n" % (env.parameters+"_"+env.model_identifier,env.parameters+"_"+env.model_identifier,env_performance_type(env),
                                                       env_performance_label(env),env.total_error,env.total_regression_error,env.total_communication, env.total_message_size))
    handle.close()
        
def output_generate_chart_script(experiment_timestamp, envs):
    timestamp_folder_path = helpers.get_timestamp_folder(experiment_timestamp)
    chart_generation_file_path = os.path.join(timestamp_folder_path, 'chart_generation.py')
    handle = open(chart_generation_file_path, 'w')
    handle.write(CHART_GENERATION_SCRIPT_TEXT)
    handle.close()
#    dirname = os.path.dirname(os.path.abspath(generate_charts_file_path))
#    return (dirname, generate_charts_file_path)
    

def env_performance_label(env):
    label = None
    if env.parameters == "serial":
        label = "Serial"
    elif env.parameters == "nosync":
        label = "No Synchronization"
    else:
        label = str(env.batchSizeInMacroRounds)
        if env.syncOperator.parameter_string!="": label+="-"+env.syncOperator.parameter_string
    
    return label

def env_performance_type(env):
    env_type = 'baseline'
    if env.parameters == "serial":
        env_type = "baseline"
    elif env.parameters == "nosync":
        env_type = "baseline"
    else:
        env_type = env.syncOperator.type    
    return env_type

#def env_performance_label(env):
#    label = None
#    if env.parameters == "serial":
#        label = "Serial"
#    elif env.parameters == "nosync":
#        label = "No Synchronization"
##    elif env.syncOperator.divergence_threshold is None:
##        label = str(env.batchSizeInMacroRounds)
#    else:
#        label = str(env.batchSizeInMacroRounds)
#        if env.syncOperator.parameter_string!="": label+="-"+env.syncOperator.parameter_string
##        label = "%d-%.2f" % (env.batchSizeInMacroRounds, env.syncOperator.divergence_threshold)
#    
#    return "'%s' : '%s'" % (env.parameters, label)
#
#
#def env_performance_type(env):
#    env_type = 'baseline'
#    if env.parameters == "serial":
#        env_type = "baseline"
#    elif env.parameters == "nosync":
#        env_type = "baseline"
#    else:
#        env_type = env.syncOperator.type    
#    return "'%s' : '%s'" % (env.parameters, env_type)

def experiment_number_of_rounds(experiment_timestamp,rounds):
    timestamp_folder_path = helpers.get_timestamp_folder(experiment_timestamp)
    summary_file_path = os.path.join(timestamp_folder_path, "summary.txt")
    summary_handle = open(summary_file_path, 'a')
    summary_handle.write(NUMBER_OF_ROUNDS_KEY+"\t%d\n" % (rounds))
    summary_handle.close()
    
def run_chart_scripts(dirname, generate_charts, performance_chart):
    print "Changing dir to: %s" % dirname
    os.chdir(dirname)    
#    command = "python \"%s\"" % generate_charts
#    if time_control.current_example <= 100000:
#        print "Running: %s" % command
#        os.system(command)
#    else:
#        print "Run this command to generate charts. Warning! This will take a long time"
#        print command
    command = "python \"%s\"" % performance_chart
    print "Running: %s" % command
    os.system(command)
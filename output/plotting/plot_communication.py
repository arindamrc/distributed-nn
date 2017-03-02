'''
Created on 20.01.2013

@author: Mario Boley
'''
import os
import matplotlib.pyplot as plt
from output.parseOutput import ENV_ID_COLUMN, ENV_LABEL_COLUMN, read_results_column, RESULT_FILENAME, read_number_of_rounds, SUMMARY_FILENAME
from output.plotting.plotting import generate_color, FILE_FORMAT
from output.plotting.plotting import save_chart, plot_drift_times
from output.colorUtils import ColorUtil

def generate(results_root_folder, show_drift_times, show_sync_times):
    print "Plotting Cumulative Communication"
    plot_chart(results_root_folder, show_drift_times,show_sync_times)
    plt.semilogy()
    plt.xlabel("Time")
    plt.ylabel("Cumulative Communication")
#    setFigLinesBW(plt.figure())
    save_chart(results_root_folder+"charts/communication_time."+FILE_FORMAT)

# Chart generation functions:
def plot_chart(results_root_folder, show_drift_times, show_sync_times):    
    envs=read_results_column(results_root_folder+RESULT_FILENAME,ENV_ID_COLUMN)
    labels=read_results_column(results_root_folder+RESULT_FILENAME,ENV_LABEL_COLUMN)
    index=-1
    colors = ColorUtil().generateDistinctColors(len(envs))
    for env in envs:
        index+=1
        file_name=results_root_folder+'logs/communication_%s.log' % (env)
        if os.path.isfile(file_name):
            metric = read_cumulative_communication(file_name,
                                               read_number_of_rounds(results_root_folder+SUMMARY_FILENAME))
            xpoints, ypoints = metric
            color = generate_color(index,len(envs))
            color = colors[index]
#             if 'static-SGD' in labels[index]:
#                 color = 'g'
#             if 'static-Kernel' in labels[index]:
#                 color = 'c'
#             if 'dynamic-SGD' in labels[index]:
#                 color = 'r'
#             if 'dynamic-Kernel' in labels[index]:
#                 color = 'm'
            plt.plot(xpoints, ypoints, label = labels[index], color = color, linewidth=2.0)
#            if not "communication" in chart:
            if show_sync_times:
                plot_sync_times(results_root_folder, env, sync_times_marker, sync_times_size, color, xpoints, ypoints)
        if show_drift_times: 
            plot_drift_times(results_root_folder)
        
def plot_sync_times(results_root_folder, env, sync_times_marker, sync_times_size, color, xpoints, ypoints):
    result_xpoints = []
    result_ypoints = []
    if os.path.isfile(results_root_folder+"logs/communication_%s.log" % env):
        sync_times = read_times_file(results_root_folder+"logs/communication_%s.log" % env)

        for idx in xrange(len(ypoints)):
            x = xpoints[idx]
            if x in sync_times:
                y = ypoints[idx]
                result_xpoints.append(x)
                result_ypoints.append(y)
                
    plt.scatter(result_xpoints, result_ypoints, s = sync_times_size, c=color, marker=sync_times_marker, alpha=0.5)
 
def read_times_file(file_name):
    result = set()
    handle = open(file_name, 'r')
    for line in handle:
        result.add(int(line.strip().split("\t")[0]))
    handle.close()
        
    return result
   
def read_cumulative_communication(file_name,number_of_rounds):
    rounds = []
    values = []
    last_round = 1
    last_value = 0.0
    current_comm = 0.0
    handle = open(file_name, 'r')
    for line in handle:
        parts = line.strip().split("\t")
        round_number = int(parts[0])
        value = float(parts[1])
        current_comm+=value
        for i in range(last_round, round_number):
            rounds.append(i)
            values.append(last_value)
        rounds.append(round_number)
        values.append(current_comm)
        last_round = round_number
        last_value = current_comm
        
    handle.close()

    if last_round < number_of_rounds:
        for idx in xrange(number_of_rounds - last_round):
            rounds.append(last_round + idx + 1)
            values.append(last_value)

    return (rounds, values)
    
#        if len(values) < number_of_rounds:
#            last_value = values[-1]
#            last_round = rounds[-1]
#            for idx in xrange(number_of_rounds - len(values)):
#                rounds.append(last_round + idx + 1)
#                values.append(last_value)

def read_sync_communication(file_name,number_of_rounds):
    rounds = []
    values = []
    current_comm = 0.0
    handle = open(file_name, 'r')
    for line in handle:
        parts = line.strip().split("\t")
        round_number = int(parts[0])
        value = float(parts[1])
        current_comm = value
        rounds.append(round_number)
        values.append(current_comm)
        
    handle.close()

    return (rounds, values)    

sync_times_marker = 'x'
sync_times_size = 60
#number_of_rounds = 100


if __name__ == "__main__":
    generate("./testdata/", True, True)
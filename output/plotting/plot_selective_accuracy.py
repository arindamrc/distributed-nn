'''
Created on 20.01.2013

@author: Mario Boley
'''
import os
import matplotlib.pyplot as plt
from output.parseOutput import read_times_file, read_results_column, ENV_ID_COLUMN, ENV_LABEL_COLUMN, RESULT_FILENAME
from output.plotting.plotting import FILE_FORMAT, save_chart, generate_color, plot_drift_times
from learning.lossFunction import ZeroOneClassificationLoss


def generate(experiment_root_folder, show_drift_times, show_sync_times, select = 1, loss_function=ZeroOneClassificationLoss()):
    print "Plotting Selective Accuracy. Select interval = " + str(select) + "Loss function: " + loss_function.name
    plot_chart(experiment_root_folder, show_drift_times, show_sync_times, select, loss_function)
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    save_chart(experiment_root_folder + "charts/selective_%d_accuracy_" % (select) + loss_function.shortname + "_time." + FILE_FORMAT)

# Chart generation functions:
def plot_chart(experiment_root_folder, show_drift_times, show_sync_times, select, loss_function):
    envs = read_results_column(experiment_root_folder + RESULT_FILENAME, ENV_ID_COLUMN)
    labels = read_results_column(experiment_root_folder + RESULT_FILENAME, ENV_LABEL_COLUMN)    
    index = -1
    max_y = 0
    max_x = 0
    for env in envs:
        index += 1
        metric = read_accuracy(experiment_root_folder + 'logs/prediction_%s.log' % (env), select, loss_function)
        write_metric(experiment_root_folder, metric)
        xpoints, ypoints = metric
        max_env_x = max(xpoints)
        max_env_y = max(ypoints)
        if max_env_x > max_x: max_x = max_env_x
        if max_env_y > max_y: max_y = max_env_y
        color = generate_color(index, len(envs))
        plt.plot(xpoints, ypoints, label=labels[index], color=color)
        if show_sync_times: plot_sync_times(experiment_root_folder, env, color, (xpoints, ypoints))
    plt.xlim(0, max_x)
    plt.ylim(0, max_y * 1.05)
    if show_drift_times: plot_drift_times(experiment_root_folder)
    
def write_metric(experiment_root_folder, metric):
    file_name = experiment_root_folder + "/charts/selective_accuracy.txt"
    rounds, values = metric
    f = open(file_name, 'w')
    for idx in xrange(len(values)):
        round_number = rounds[idx]
        value = values[idx]
        f.write("%d\t%.5f\n" % (round_number, value))
    f.close()  
        
        
def plot_sync_times(experiment_root_folder, env, color, metric):
    result_xpoints = []
    result_ypoints = []
    xpoints, ypoints = metric
    if os.path.isfile(experiment_root_folder + "logs/communication_%s.log" % env):
        sync_times = read_times_file(experiment_root_folder + "logs/communication_%s.log" % env)

        for idx in xrange(len(ypoints)):
            x = xpoints[idx]
            if x in sync_times:
                y = ypoints[idx]
                result_xpoints.append(x)
                result_ypoints.append(y)
                
    plt.scatter(result_xpoints, result_ypoints, s=sync_times_size, c=color, marker=sync_times_marker)

def read_accuracy(file_name, select, loss_function):
    rounds = []
    values = []
    currentError = 0.0
    currentRound = 0
    handle = open(file_name, 'r')
    index = 0
    considered = 0
    for line in handle:
        parts = line.strip().split("\t")
        round_number = int(parts[0])
        if index % select == 0:
            considered += 1
            if round_number > currentRound:        
                rounds.append(round_number)
                values.append(1.0 - float(currentError) / considered)
                currentRound = round_number
            currentError += loss_function(float(parts[1]), float(parts[2]))
        index += 1
#        if float(parts[1])!=signum(float(parts[2])): currentError+=1
        
    handle.close()
    return (rounds, values)


show_drift_times = True
show_sync_times = True
sync_times_marker = 'x'
sync_times_size = 60

if __name__ == "__main__":
    generate("./testdata/", show_drift_times, show_sync_times)

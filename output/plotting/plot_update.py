'''
Created on 20.01.2013

@author: Mario Boley
'''
import os
import matplotlib.pyplot as plt
from output.parseOutput import read_times_file, read_results_column, ENV_ID_COLUMN, ENV_LABEL_COLUMN, RESULT_FILENAME
from output.plotting.plotting import FILE_FORMAT, save_chart, generate_color, plot_drift_times


def generate(experiment_root_folder):
    print "Plotting Cumulative Update"
    plot_chart(experiment_root_folder)
    plt.xlabel("Time")
    plt.ylabel("Update")
    save_chart(experiment_root_folder+"charts/update_time."+FILE_FORMAT)

# Chart generation functions:
def plot_chart(experiment_root_folder):
    envs=read_results_column(experiment_root_folder+RESULT_FILENAME,ENV_ID_COLUMN)
    labels=read_results_column(experiment_root_folder+RESULT_FILENAME,ENV_LABEL_COLUMN)    
    index=-1
    max_y = 0
    max_x = 0
    for env in envs:
        index+=1
        metric = read_cumulative_update(experiment_root_folder+'logs/update_%s.log' % (env))
        xpoints, ypoints = metric
        max_env_x = max(xpoints)
        max_env_y = max(ypoints)
        if max_env_x > max_x: max_x=max_env_x
        if max_env_y > max_y: max_y=max_env_y
        color = generate_color(index,len(envs))
        plt.plot(xpoints, ypoints, label = labels[index], color = color)
        if show_sync_times: plot_sync_times(experiment_root_folder, env, color, (xpoints, ypoints))
    plt.xlim(0,max_x)
    plt.ylim(0,max_y*1.05)
    if show_drift_times: plot_drift_times(experiment_root_folder)
        
        
def plot_sync_times(experiment_root_folder, env, color, metric):
    result_xpoints = []
    result_ypoints = []
    xpoints, ypoints = metric
    if os.path.isfile(experiment_root_folder+"logs/communication_%s.log" % env):
        sync_times = read_times_file(experiment_root_folder+"logs/communication_%s.log" % env)

        for idx in xrange(len(ypoints)):
            x = xpoints[idx]
            if x in sync_times:
                y = ypoints[idx]
                result_xpoints.append(x)
                result_ypoints.append(y)
                
    plt.scatter(result_xpoints, result_ypoints, s = sync_times_size, c=color, marker=sync_times_marker)

def read_cumulative_update(file_name):
    rounds = []
    values = []
    current_update=0.0
    currentRound=0
    handle = open(file_name, 'r')
    for line in handle:
        parts = line.strip().split("\t")
        round_number = int(parts[0])
        if round_number>currentRound:
            rounds.append(round_number)
            values.append(current_update)
            currentRound=round_number
        current_update+= float(parts[1])
#        if float(parts[1])!=signum(float(parts[2])): current_update+=1
        
    handle.close()
    return (rounds, values)


show_drift_times = True
show_sync_times = True
sync_times_marker = 'x'
sync_times_size = 60

if __name__ == "__main__":
    generate("./testdata/", show_drift_times, show_sync_times)
'''
Created on 20.01.2013

@author: Mario Boley
'''

import matplotlib.pyplot as plt
from output.colorUtils import ColorUtil
from output.parseOutput import read_results_column, RESULT_FILENAME, ENV_ID_COLUMN, ENV_ERROR_COLUMN, ENV_LABEL_COLUMN, ENV_TYPE_COLUMN, ENV_PARAM_COLUMN, ENV_COMM_MESSAGE_COLUMN
from output.plotting.plotting import CHART_DIMENSIONS, FILE_FORMAT, save_chart
import matplotlib
from statsmodels.base.model import Results

def generate(result_root_folder,predictive_performance_column=ENV_ERROR_COLUMN, plotBaseLines = True, loss_function = None):
    print "Plotting Performance"
    result_filename=result_root_folder+RESULT_FILENAME
    envs = read_results_column(result_filename,ENV_ID_COLUMN)
    labels = read_results_column(result_filename, ENV_LABEL_COLUMN)
    types = read_results_column(result_filename,ENV_TYPE_COLUMN)
    _ = read_results_column(result_filename,ENV_PARAM_COLUMN)
    values = read_metrics(envs,result_filename,predictive_performance_column)
    #if loss_function != None:
    #    for index in xrange(len(envs)):
    #        cumError = read_cumulative_error(result_root_folder+'logs/prediction_%s.log' % (envs[index]),loss_function)
    #        values[index] = (values[index][0],cumError)
    
#    matplotlib.rcParams.update({'font.size': 16})
#    matplotlib.rc('xtick', labelsize=12) 
#    matplotlib.rc('ytick', labelsize=12) 
    matplotlib.rcParams.update({'axes.labelsize': 14})
    
    plot_points(values, labels, types, markers_size)
    if plotBaseLines:
        plot_baselines(types,result_filename,predictive_performance_column)    
        #   set_plot_axes(x_axis, y_axis)
        #plot_baselines_annotation(envs, types, labels,result_filename,predictive_performance_column)
    
    plt.title(title)
    
    x_label = "Cumulative Communication"
    y_label = "Cumulative Error"
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(-3000,88000)      
    #plt.legend(loc='upper left')
    
    fig = plt.gcf()
    fig.set_size_inches(*CHART_DIMENSIONS)
    
#     plt.savefig(result_root_folder+'charts/performance.%s' % (FILE_FORMAT), 
#                 format=FILE_FORMAT, dpi = CHART_DPI, bbox_inches='tight')
    
    save_chart(result_root_folder+'charts/performance.%s' % (FILE_FORMAT))
    
    plt.clf()
    
def read_metrics(envs,result_filename,predictive_performance_column):
    result = []
    errors = read_results_column(result_filename, predictive_performance_column)
    communication = read_results_column(result_filename, ENV_COMM_MESSAGE_COLUMN)
    for i in xrange(len(envs)):
        result.append((float(communication[i]), float(errors[i])))
    print envs
    print result
    print errors    
    return result

def read_cumulative_error(file_name,loss_function):
    rounds = []
    values = []
    currentError=0.0
    currentRound=0
    handle = open(file_name, 'r')
    for line in handle:
        parts = line.strip().split("\t")
        currentError+=loss_function(float(parts[1]),float(parts[2]))
#        if float(parts[1])!=signum(float(parts[2])): currentError+=1
    return currentError

def plot_baselines(types, result_file,predictive_performance_column):
    baseline_markers = ['--', '-.', '-',':','--', '-.', '-',':']
    baseline_color = 'k'

    marker_index = 0    
    for i in xrange(len(types)):
        env_type = types[i]
        if env_type == 'baseline':
            marker = baseline_markers[marker_index]
            marker_index += 1
            y = read_results_column(result_file,predictive_performance_column)[i]
            name  = read_results_column(result_file,1)[i]
            plt.axhline(y, color = baseline_color, linestyle = marker, label=r''+name)
            
def plot_baselines_annotation(envs, types, labels, result_file,predictive_performance_column):
    _,xmax,_,_ = plt.axis()
    x_annotation_position = 0.9 * xmax
    for i in xrange(len(envs)):
        env_type = types[i]
        if env_type == 'baseline':
            y = read_results_column(result_file,predictive_performance_column)[i]
            annotation = read_results_column(result_file,ENV_LABEL_COLUMN)[i]
            plt.annotate(annotation, xy = (x_annotation_position, y), xytext = (0, 15),
                        textcoords = 'offset points', ha = 'right', va = 'top')

def plot_points(values, labels, types, markers_size):
#     scatters = {}
#     max_x = 0
#     for i in xrange(len(types)):
#         if types[i] != 'baseline':
#             x, y = values[i]
#             if x>max_x: max_x=x
#             x_points, y_points = scatters.get(types[i], ([], []))
#             x_points.append(x)
#             y_points.append(y)
#             scatters[types[i]] = (x_points, y_points)
#     
#     i=0
#     for env_type, (x_points, y_points) in scatters.items():
#         marker = MARKERS[i%len(MARKERS)]
# #        marker = markers[env_type]
#         color = COLORS[i%len(COLORS)]
#         plt.scatter(x_points, y_points, s = markers_size, color = color, marker = marker, label = labels[i])
#         print labels[i]
#         i+=1
    colors = ColorUtil().generateDistinctColors(len(values))
    for i in xrange(len(values)):
        if types[i] != 'baseline':
            marker = MARKERS[i%len(MARKERS)]
            color = colors[i]
            x,y = values[i]
            plt.scatter(x, y, s = markers_size, color = color, marker = marker, label = r''+labels[i], alpha = 0.7)
    
    #plt.legend(loc='upper right')
    
#     for i in xrange(len(values)):
#         (x, y) = values[i]
#         env_type = types[i]
#         if env_type != 'baseline':
#             annotation = labels[i]
#             plt.annotate(annotation, xy = (x, y), xytext = (len(annotation) * 3, -5),
#                          textcoords = 'offset points', ha = 'right', va = 'top')

    #plt.xlim(xmin=0)
#    plt.yscale('log')
#    plt.xscale('log')
    #plt.xlim(xmax=5000000)
#def set_plot_axes(x_axis, y_axis):
#    xmin,xmax,ymin,ymax = plt.axis()
#    y1, y2 = ymin, ymax
#    plt.axis([xmin, xmax, y1, y2])
    

title =''

# Environments types markers:
#markers = {
#            'active' : 'x',
#'hedge' : 'o',
#'static' : 's',
#'nosync' : '^',
#'baseline':['--', '-.', '-']
#           }

MARKERS = ['x','o','s','^','v','*','p','+']
COLORS = ['0.4','0.3','0.6']
COLOR = ['RED', 'BLUE', 'GREEN','YELLOW']

markers_size = 400

# Environments types colors:
#markers_color = {
#'active' : '0.4',
#'hedge' : '0.3',
#'static' : '0.6',
#'nosync' : '0.4',
#'baseline':'k'
#                 }
## Axis values (None means automatic) should go from low value to high value (in plot they will be reversed)
#x_axis = None
#y_axis = None

# Legend location
legend_location = "center right"

if __name__ == "__main__":
    generate("./testdata/")

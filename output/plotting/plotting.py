'''
Created on 20.01.2013

@author: Mario Boley
'''
import matplotlib.pyplot as plt
from output.parseOutput import read_times_file
import os

from matplotlib.cm import get_cmap
import math
import matplotlib

#CHART_DIMENSIONS = (10, 10)
golden_mean = (math.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width=10
##fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
CHART_DIMENSIONS = [fig_width,fig_height]
CHART_DPI = 500
FILE_FORMAT = "png"#"pdf"
COLOR_MAP = get_cmap('jet')

matplotlib.rcParams['agg.path.chunksize'] = 10000

def generate_color(index, total):
    return COLOR_MAP(1. * index / total)

def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    COLORMAP = {
        'b': {'marker': None, 'dash': (None,None)},
        'g': {'marker': None, 'dash': [5,5]},
        'r': {'marker': None, 'dash': [5,3,1,3]},
        'c': {'marker': None, 'dash': [1,3]},
        'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
        'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }

    for line in ax.get_lines():
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)

def save_chart(filename):   
    box = plt.axes().get_position()
    plt.axes().set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # Put a legend to the right of the current axis
    plt.axes().legend(loc='center left', bbox_to_anchor=(0.485, 0.15), markerscale = 0.35)     
    #plt.legend(loc = "upper left", fontsize = 8)
        
    fig = plt.gcf()
    fig.set_size_inches(*CHART_DIMENSIONS)

#    plt.axes().set_xlim(x_axis)
#    else:
#        plt.axes().set_xlim((0, number_of_rounds))
#    ymin, ymax = plt.axes().get_ylim()
#    plt.axes().set_ylim(y_axis)
#    else:
#        plt.axes().set_ylim((0.0, ymax))
        
    plt.savefig(filename, FILE_FORMAT=FILE_FORMAT, dpi = CHART_DPI, bbox_inches='tight')
    
    plt.clf()
    
def plot_drift_times(results_root_path):
    if os.path.isfile(results_root_path+"logs/drift_global_.log"):
        for drift_time in read_times_file(results_root_path+"logs/drift_global_.log"):
            plt.axvline(drift_time, color = "0.3")
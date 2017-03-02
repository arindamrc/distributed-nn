'''
Created on 20.01.2013

@author: Mario Boley
'''
import os

RESULT_FILENAME = "results.txt"

ENV_ID_COLUMN = 0
ENV_LABEL_COLUMN = 1
ENV_TYPE_COLUMN = 2
ENV_PARAM_COLUMN = 3
ENV_ERROR_COLUMN = 4
ENV_REGRESSION_ERROR_COLUMN = 5
ENV_COMM_COLUMN = 6
ENV_COMM_MESSAGE_COLUMN = 7

SUMMARY_FILENAME = "summary.txt"

NUMBER_OF_ROUNDS_KEY = "rounds:"

def read_results_column(result_file,column_idx):
    column = []
    if os.path.isfile(result_file):
        handle = open(result_file, 'r')
        for line in handle:
            stripped_line = line.strip()
            if line.startswith("#") or len(stripped_line)==0:
                continue
            tokens=stripped_line.split("\t")
            column.append(tokens[column_idx])
                
        handle.close()
    return column

def read_number_of_rounds(summary_file):
    handle = open(summary_file, 'r')
    for line in handle:
        stripped_line = line.strip()
        tokens=stripped_line.split("\t")
        if tokens[0]==NUMBER_OF_ROUNDS_KEY:
            return int(tokens[1])
    raise KeyError
        
#def read_environments(list_of_envs_file):
#    all_environments = []
#    if os.path.isfile(list_of_envs_file):
#        handle = open(list_of_envs_file, 'r')
#        for line in handle:
#            stripped_line = line.strip()
#            if line.startswith("#"):
#                continue
#            if len(stripped_line)>0:
#                tokens=stripped_line.split("\t")
#                all_environments.append(tokens[0])
#                
#        handle.close()
#    return all_environments
#
#def read_labels(list_of_envs_file):
#    all_environments = []
#    if os.path.isfile(list_of_envs_file):
#        handle = open(list_of_envs_file, 'r')
#        for line in handle:
#            stripped_line = line.strip()
#            if line.startswith("#"):
#                continue
#            if len(stripped_line)>0:
#                tokens=stripped_line.split("\t")
#                all_environments.append(tokens[1])
#                
#        handle.close()
#    return all_environments

def read_times_file(file_name):
    result = set()
    handle = open(file_name, 'r')
    for line in handle:
        result.add(int(line.strip().split("\t")[0]))
    handle.close()
        
    return result

from time import strftime
import os
timestamp = strftime("%d.%m.%Y_%H_%M_%S")
RESULT_FILE = "Results_" + timestamp + ".csv"

PATH_TO_RESULTS = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../results/") + "/"
PATH_TO_DATASETS = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/data/") + "/"
PATH_TO_TWITTER = os.path.join(PATH_TO_DATASETS, 'TwitterPreprocessed')
PATH_TO_FINANCIAL = os.path.join(PATH_TO_DATASETS, 'financial')
PATH_TO_FINANCE = os.path.join(PATH_TO_DATASETS, 'finance')
PATH_TO_FINANCIAL_NEWS = os.path.join(PATH_TO_DATASETS, 'financial_news')

def getPathToDataset(datasetName):
    return PATH_TO_DATASETS + datasetName + '/' + datasetName + '.dat'

def get_path_to_uci_dataset(name):
    return PATH_TO_DATASETS + 'uci/' + '%s.data' % (name)

if __name__=="__main__":
    print PATH_TO_TWITTER

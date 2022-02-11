import argparse
from process import DATA_PROCESS

# parser initialization
parser = argparse.ArgumentParser(description='Script for datasets preprocessing')
parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
args = parser.parse_args()
dataset = args.dataset 

if dataset in DATA_PROCESS.keys():
    DATA_PROCESS[dataset]()
else:
    print('This dataset is not handled for the moment')

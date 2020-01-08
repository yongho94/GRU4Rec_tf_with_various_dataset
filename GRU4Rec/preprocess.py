import argparse
from utils import *
import os

DATASETDIR = './../dataset'
DATASET_handler = {
        'movie':['ratings_Movies20M.csv'],
        'books':['ratings_Books.csv']
    }

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--minSessionLen', type=int, default=5)

args = parser.parse_args()

def check_args(args):
    print(" * Check arguments")

    assert args.dataset is not None, "please type python preprocess.py --dataset [dataset name]"
    dataset_loc = os.path.join(DATASETDIR, args.dataset)
    assert os.path.exists(dataset_loc), "Couldn't find dataset file at " + dataset_loc

    dataset_proc = None
    for proc in DATASET_handler:
        if args.dataset in DATASET_handler[proc]:
            dataset_proc = proc
    assert dataset_loc is not None, "There is no preprocessor for " + args.dataset

    return dataset_proc

def preprocess(args, dataset_proc):
    print(" * preprocess", args.dataset, "dataset")
    if dataset_proc == 'movie':
        prepro_movie(args)
    elif dataset_proc == 'books':
        prepro_books(args)
    else:
        print('No preprocessor Error Occured')
        exit(0)

if __name__ == '__main__':
    dataset_proc = check_args(args)
    preprocess(args, dataset_proc)

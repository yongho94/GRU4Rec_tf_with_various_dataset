import argparse
from utils import *
import os

DATASET_handler = {
    'movie': ['ratings_Movies20M.csv'],
    'books': ['ratings_Books.csv'],
    'rsc15': ['yoochoose-clicks.dat']
}

parser = argparse.ArgumentParser()
parser.add_argument('--raw_loc', type=str, default='raw_dataset/', help='raw dataset location')
parser.add_argument('--dataset', type=str, default=None, help='raw dataset name')
parser.add_argument('--minSessionLen', type=int, default=5)
parser.add_argument('--minFreq', type=int, default=5)

args = parser.parse_args()

def check_args(args):
    print(" * Check arguments")

    assert args.dataset is not None, "please type python preprocess.py --dataset [dataset name]"
    dataset_proc = None
    for proc in DATASET_handler:
        if args.dataset in DATASET_handler[proc]:
            dataset_proc = proc
    assert dataset_proc is not None, "There is no preprocessor for " + args.dataset

    return dataset_proc


def preprocess(args, dataset_proc):
    print(" * Preprocess", args.dataset, "dataset")
    if dataset_proc == 'movie':
        prepro_movie(args)
    elif dataset_proc == 'books':
        prepro_books(args)
    elif dataset_proc == 'rsc15':
        prepro_rsc15(args)
    else:
        print('There is No preprocessor for', args.dataset)
        exit(0)

if __name__ == '__main__':
    dataset_proc = check_args(args)
    preprocess(args, dataset_proc)

import os
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import *
from sklearn.utils import shuffle

def truncate_data(args, data):
    print(" * Original Number of data :", len(data))

    # truncated short sessions
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print(" * Truncated data by session", args.minSessionLen, ":", len(data))

    # truncated insufficiently occured movies
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= args.minFreq].index)]
    print(" * Truncated data by freq", args.minFreq, ':', len(data))

    # truncated short sessions
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print(" * Truncated data by session", args.minSessionLen, ":", len(data))

    return data

def save_data(args, data):
    print("\n * Total Data event :", len(data),
          "\n * Num of Total SessionId(User) :", len(data.groupby('SessionId').size()),
          "\n * Num of Total Item :", len(data.groupby('ItemId').size()))
    dataset_name = ''.join(args.dataset.split('.')[:-1]) + '.txt'
    data = data.sort_values(['SessionId', 'Time'])
    data.to_csv("prcd_"+dataset_name, sep='\t', index=False)
    print(' * Done')
    return

def prepro_movie(args):
    data = pd.read_csv(args.raw_loc+args.dataset, sep=',', header=0, dtype={0: np.int32, 1: np.int32, 2: np.float64, 3: np.int32})
    new_columns = {'userId': 'SessionId', 'movieId': 'ItemId', 'timestamp': 'Time'}
    data = data.rename(columns=new_columns)[['SessionId', 'ItemId', 'Time']]
    data = truncate_data(args, data)
    save_data(args, data)
    return

def prepro_books(args):
    data = pd.read_csv(args.raw_loc+args.dataset, sep=',', header=None,
                       names=['userId', 'bookId', 'rating', 'timestamp'],
                       dtype={0: 'category', 1: 'category', 2: np.float64, 3: np.int32})
    data['k_userId'] = data['userId'].cat.codes
    data['k_bookId'] = data['bookId'].cat.codes
    data = data.rename(columns={'k_userId': 'SessionId', 'k_bookId': 'ItemId', 'timestamp': 'Time'})
    data = data[['SessionId', 'ItemId', 'Time']]
    data = truncate_data(args, data)
    save_data(args, data)
    return

def prepro_rsc15(args):
    data = pd.read_csv(args.raw_loc+args.dataset, sep=',', header=None, usecols=[0,1,2],
                       dtype={0:np.int32, 1:str, 2:np.int64})
    data.columns = ['SessionId', 'TimeStr', 'ItemId']
    data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    del (data['TimeStr'])
    data['Time'] = data['Time'] * 1000
    data['Time'] = data['Time'].astype(np.int64)
    data = data[['SessionId', 'ItemId', 'Time']]
    data = truncate_data(args, data)
    save_data(args, data)
    return

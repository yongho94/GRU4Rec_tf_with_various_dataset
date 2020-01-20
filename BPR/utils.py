import os
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import *
from sklearn.utils import shuffle

def prepro_movie(args, dataset_loc):
    data = pd.read_csv(dataset_loc, sep=',', header=0, dtype={0: np.int32, 1: np.int32, 2: np.float64, 3: np.int32})
    print(" * Original Number of data :", len(data))
    session_lengths = data.groupby('userId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print(" * Truncated data by session", args.minSessionLen, ":", len(data))

    movie_supports = data.groupby('movieId').size()
    # truncated insufficiently occured movies
    data = data[np.in1d(data.movieId, movie_supports[movie_supports >= args.minFreq].index)]
    print(" * Truncated data by freq", args.minFreq, ':', len(data))

    session_lengths = data.groupby('userId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print(" * Truncated data by session", args.minSessionLen, ":", len(data))

    new_columns = {'userId': 'SessionId', 'movieId': 'ItemId'}
    data = data.rename(columns=new_columns)
    data = shuffle(data)[['SessionId', 'ItemId']]

    train_len = int(len(session_lengths) * args.train_ratio)
    valid_len = int(len(session_lengths) * args.valid_ratio)

    train_sess = session_lengths[:train_len]
    valid_sess = session_lengths[train_len:train_len + valid_len]
    test_sess = session_lengths[train_len + valid_len:]

    train_data = data[np.in1d(data.SessionId, train_sess.index)]
    valid_data = data[np.in1d(data.SessionId, valid_sess.index)]
    test_data = data[np.in1d(data.SessionId, test_sess.index)]

    print("Training data length:", len(train_data),
          "\nValidation data lengths:", len(valid_data),
          "\nTest data length:", len(test_data))

    train_data.to_csv('./data/' + args.dataset[:-4] + '_train.txt', sep='\t', index=False)
    valid_data.to_csv('./data/' + args.dataset[:-4] + '_valid.txt', sep='\t', index=False)
    test_data.to_csv( './data/' + args.dataset[:-4] + '_test.txt', sep='\t', index=False)

    print(' * Done')
    return


def prepro_books(args, dataset_loc):
    data = pd.read_csv(dataset_loc, sep=',', header=None,
                       names=['userId', 'bookId', 'rating', 'timestamp'],
                       dtype={0: 'category', 1: 'category', 2: np.float64, 3: np.int32})
    data['k_userId'] = data['userId'].cat.codes
    data['k_bookId'] = data['bookId'].cat.codes
    data = data[['k_userId', 'k_bookId']]
    data = data.rename(columns={'k_userId': 'SessionId', 'k_bookId': 'ItemId', 'timestamp': 'Time'})

    print("Original Number of data :", len(data))
    session_lengths = data.groupby('SessionId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print("Truncated data by session", args.minSessionLen, ":", len(data))

    item_supports = data.groupby('ItemId').size()
    # truncated insufficiently occured movies
    data = data[np.in1d(data.itemId, item_supports[item_supports >= args.minFreq].index)]
    print("Truncated data by freq", args.minFreq, ':', len(data))

    session_lengths = data.groupby('SessionId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print("Truncated data by session", args.minSessionLen, ":", len(data))

    data = shuffle(data)[['SessionId', 'ItemId']]

    train_len = int(len(session_lengths) * args.train_ratio)
    valid_len = int(len(session_lengths) * args.valid_ratio)

    train_sess = session_lengths[:train_len]
    valid_sess = session_lengths[train_len:train_len + valid_len]
    test_sess = session_lengths[train_len + valid_len:]

    train_data = data[np.in1d(data.SessionId, train_sess.index)]
    valid_data = data[np.in1d(data.SessionId, valid_sess.index)]
    test_data = data[np.in1d(data.SessionId, test_sess.index)]

    print("Training data length:", len(train_data),
          "\nValidation data lengths:", len(valid_data),
          "\nTest data length:", len(test_data))

    train_data.to_csv('./data/' + args.dataset[:-4] + '_train.txt', sep='\t', index=False)
    valid_data.to_csv('./data/' + args.dataset[:-4] + '_valid.txt', sep='\t', index=False)
    test_data.to_csv('./data/' + args.dataset[:-4] + '_test.txt', sep='\t', index=False)

    print(' * Done')
    return
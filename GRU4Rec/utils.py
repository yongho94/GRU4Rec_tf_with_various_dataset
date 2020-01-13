import os
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import *

def prepro_movie(args, dataset_loc):
    
    data = pd.read_csv(dataset_loc, sep=',', header=0, dtype={0:np.int32, 1:np.int32, 2:np.float64, 3:np.int32})
    print(" * Original Number of data :", len(data))
    session_lengths = data.groupby('userId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print(" * Truncated data by session", args.minSessionLen,":",len(data))
    
    movie_supports = data.groupby('movieId').size()
    # truncated insufficiently occured movies
    data = data[np.in1d(data.movieId, movie_supports[movie_supports >= args.minFreq].index)]
    print(" * Truncated data by freq", args.minFreq,':', len(data))

    session_lengths = data.groupby('userId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print(" * Truncated data by session", args.minSessionLen,":",len(data))
    
    data = data.sort_values(by=['userId', 'timestamp'], axis=0)
    train_lengths = (session_lengths * args.train_ratio).astype(int)
    
    train_mask = []
    for idx in tqdm(session_lengths.index, "Spliting data"):
        train_mask += [True] * train_lengths[idx] 
        train_mask += [False] * (session_lengths[idx] - train_lengths[idx])
    
    train_data = data[train_mask]
    train_session_lengths = train_data.groupby('userId').size()
    train_data = train_data[
        np.in1d(train_data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    
    train_data = train_data[['userId','movieId','timestamp']]
    
    new_columns = {'userId':'SessionId', 'movieId':'ItemId', 'timestamp':'Time'}
    train_data = train_data.rename(columns=new_columns)

    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_data), train_data.SessionId.nunique(), train_data.ItemId.nunique()))
    
    vt_data = data
    vt_data = vt_data[['userId','movieId','timestamp']]
    vt_data = vt_data.rename(columns=new_columns)

    train_data.to_csv('./data/'+args.dataset[:-4]+'_train.txt', sep='\t', index=False)
    vt_data.to_csv('./data/'+args.dataset[:-4]+'_valid_test.txt', sep='\t', index=False)

    print(' * Done')
    return

def prepro_books(args, dataset_loc):
    
    data = pd.read_csv(dataset_loc, sep=',',header=None, 
        names=['userId', 'bookId', 'rating', 'timestamp'], 
        dtype={0:'category', 1:'category', 2:np.float64, 3:np.int32})
    data['k_userId'] = data['userId'].cat.codes
    data['k_bookId'] = data['bookId'].cat.codes
    data = data[['k_userId', 'k_bookId', 'timestamp']]
    data = data.rename(columns={'k_userId':'userId', 'k_bookId':'itemId','timestamp':'Time'})
    
    print("Original Number of data :", len(data))
    session_lengths = data.groupby('userId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print("Truncated data by session", args.minSessionLen,":",len(data))
    
    item_supports = data.groupby('itemId').size()
    # truncated insufficiently occured movies
    data = data[np.in1d(data.itemId, item_supports[item_supports >= args.minFreq].index)]
    print("Truncated data by freq",args.minFreq, ':', len(data))
    
    session_lengths = data.groupby('userId').size()
    # truncated short sessions
    data = data[np.in1d(data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    print("Truncated data by session", args.minSessionLen,":",len(data))
    
    data = data.sort_values(by=['userId', 'Time'], axis=0)
    session_lengths = data.groupby('userId').size()
    train_lengths = (session_lengths * args.train_ratio).astype(int)
    
    train_mask = []
    for idx in tqdm(session_lengths.index, "Splitting data"):
        train_mask += [True] * train_lengths[idx] 
        train_mask += [False] * (session_lengths[idx] - train_lengths[idx])
        
    train_data = data[train_mask]
    train_session_lengths = train_data.groupby('userId').size()
    train_data = train_data[
        np.in1d(train_data.userId, session_lengths[session_lengths >= args.minSessionLen].index)]
    
    vt_data = data
    train_data.to_csv('./data/'+args.dataset[:-4]+'_train.txt', sep='\t', index=False)
    vt_data.to_csv('./data/'+args.dataset[:-4]+'_valid_test.txt', sep='\t', index=False)
    
    print(' * Done')
    return
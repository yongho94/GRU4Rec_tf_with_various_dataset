import pandas as pd
from model import Model
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def train(args):
    print(' * Train', args.dataset)

    print(' * Loading Dataset......')
    data = pd.read_csv(os.path.join(args.data_dir, args.dataset),
            sep='\t', dtype={0:np.int64, 1:np.int64, 2:np.int64})
    data, n_items = make_ItemIdx(data)
    data = data.sort_values(['SessionId', 'Time'])
    offset_sessions = np.zeros(data['SessionId'].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = data.groupby('SessionId').size().cumsum()
    
    print(' * Loading Model......')
    model = Model(args, n_items, is_train=True)

    print(' * Start training...!')
    sess_config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(args.log_dir)
        
        for epoch in range(args.n_epochs):  # Epoch
            print("#", epoch, "Epoch Start....")
            offset_data = get_offset_data(args, offset_sessions)
            tr_start_offset, tr_end_offset = offset_data['tr']
            va_start_offset, va_end_offset, va_chk_offset = offset_data['va']
            
            iters = np.arange(args.batch_size)
            max_iter = iters.max()
            gru_state = [np.zeros([args.batch_size, args.hidden_size], dtype=np.float32) for _ in range(args.rnn_layers)]
            
            step = 20000#int( (len(data) * args.train_ratio) / args.batch_size)  
            out_idx = data.ItemIdx.values[ tr_start_offset[iters] ]
            for _ in tqdm(range(step)):
                in_idx = out_idx
                tr_start_offset[iters] += 1
                out_idx = data.ItemIdx.values[ tr_start_offset[iters]]
                feed_dict = {model.x : in_idx, model.y : out_idx, 
                             model.start : tr_start_offset[iters], model.end : tr_end_offset[iters]}
                for i in range(args.rnn_layers):
                    feed_dict[model.state[i]] = gru_state[i]
                fetches = [model.sampled_cost, model.final_state, model.mask, model.mask_num, model.train_op]
                cost, state, mask, mask_num, _ = sess.run(fetches, feed_dict)
                print(cost)
                #print(cost)
                if mask_num != 0:
                    iters[mask] = max_iter + 1 + np.arange(mask_num)
                    max_iter = iters.max()
                    for i in range(args.rnn_layers):
                        gru_state[i][mask] = 0
                
                
            print("#", epoch, "Epoch End. Testing Validation...")
            
            sampling = np.zeros([len(va_start_offset)], dtype=np.int32)
            sampling[:args.valid_num] = 1
            print(np.sum(sampling))
            sampling = sampling.astype(np.bool)
            #Shuffling must be added
            va_start_offset = va_start_offset[sampling]
            va_end_offset = va_end_offset[sampling]
            va_chk_offset = va_chk_offset[sampling]
            sess.run(tf.assign(self.model.dropout_rate, tf.constant(1, dtype=tf.float32)))
            test_model(args, data, model, sess, va_start_offset, va_end_offset, va_chk_offset, is_valid=True)
            sess.run(tf.assign(self.model.dropout_rate, tf.constant(args.dropout, dtype=tf.float32)))
            # state initialize
            # 몇번의 batch 를 줄 지를 random 으로 결정하여, N * batch_size 개수를 주고 다할 떄까지.
            # 결과는 똑같이 저장하고 있는다. 
            
    return

def test_model(args, data, model, sess, start_offset, end_offset, chk_offset, is_valid=False):
    return
    iters = np.arange(args.batch_size)
    max_iter = iters.max()
    gru_state = [np.zeros([args.batch_size, args.hidden_size], dtype=np.float32) for _ in range(args.rnn_layers)]
    out_idx = data.ItemIdx.values[ start_offset[iters] ]
    
    recall = 0
    total = 0
    
    sess_len = int( (np.sum(end_offset - start_offset) - args.batch_size) / args.batch_size)
    print(sess_len)
    for _ in tqdm(range(sess_len)):
        in_idx = out_idx
        #print("in_idx", in_idx)
        start_offset[iters] += 1
        out_idx = data.ItemIdx.values[ start_offset[iters]]
        #print("out_idx", out_idx)
        feed_dict = {model.x : in_idx, model.y : out_idx, 
                     model.start : start_offset[iters], model.end : end_offset[iters],
                     model.chk : chk_offset[iters]}
        
        for i in range(args.rnn_layers):
            feed_dict[model.state[i]] = gru_state[i]
            
        if is_valid:
            fetches = [model.cost, model.yhat, model.final_state, model.mask, model.mask_num, model.chk_mask]
            cost, yhat, state, mask, mask_num, chk_mask = sess.run(fetches, feed_dict)
        else:
            fetches = [model.yhat, model.final_state, model.mask, model.mask_num, model.chk_mask]
            yhat, state, mask, mask_num, chk_mask = sess.run(fetches, feed_dict)
            
        if chk_mask.sum() != 0:
            yhat__ = yhat[chk_mask]
            out_idx__ = out_idx[chk_mask]
            t_recall, t_total = calculate_recall(yhat__, out_idx__)
            print('recall now :', t_recall / t_total) 
            
        if mask_num != 0:
            iters[mask] = max_iter + 1 + np.arange(mask_num)
            max_iter = iters.max()
            for i in range(args.rnn_layers):
                gru_state[i][mask] = 0
            
def test(args):
    print(' * Test', args.dataset)
    return

def get_offset_data(args, offset_sessions):
    offset_sessions = np.array(offset_sessions)
    offset_len = offset_sessions[1:] - offset_sessions[:-1]

    tr_start_offset = offset_sessions[:-1]
    tr_end_offset = tr_start_offset + (offset_len * args.train_ratio).astype(np.int32)

    va_start_offset = tr_start_offset
    va_end_offset = va_start_offset + (offset_len * (args.train_ratio+args.valid_ratio)).astype(np.int32)
    va_chk_offset = tr_end_offset

    te_start_offset = va_start_offset
    te_end_offset = offset_sessions[1:]
    te_chk_offset = va_end_offset
    
    result = { 'tr' : (tr_start_offset, tr_end_offset),
               'va' : (va_start_offset, va_end_offset, va_chk_offset ),
               'te' : (te_start_offset, te_end_offset, te_chk_offset ) }
    
    return result

def make_ItemIdx(data):
    itemids = data['ItemId'].unique()
    n_items = len(itemids)
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids)
    data = pd.merge(data, pd.DataFrame({'ItemId': itemids, 'ItemIdx': itemidmap[itemids].values}),
                    on='ItemId', how='inner')
    return data, n_items

def calculate_recall(yhat, ylabel):
    
    recall = 0
    total = 0
    top20 = yhat.argsort(axis=1)[:,-20:]
    for idx, label in enumerate(ylabel):
        if label in top20[idx]:
            recall += 1
        total += 1
    
    return recall, total
            
    
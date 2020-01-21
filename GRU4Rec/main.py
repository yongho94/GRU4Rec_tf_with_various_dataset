import pandas as pd
from model import Model
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import time
from pprint import pprint

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
            pprint(offset_data)
            tr_start_offset, tr_end_offset = offset_data['tr']
            va_start_offset, va_end_offset, va_chk_offset = offset_data['va']
            
            iters = np.arange(args.batch_size)
            max_iter = iters.max()
            max_sess = len(tr_start_offset) - 1
            gru_state = [np.zeros([args.batch_size, args.hidden_size], dtype=np.float32) for _ in range(args.rnn_layers)]
            
            step = int((len(data) * args.train_ratio) / args.batch_size)
            for _ in tqdm(range(step), "Approximate Step length"):
                in_idx = data.ItemIdx.values[ tr_start_offset[iters] ]
                tr_start_offset[iters] += 1
                out_idx = data.ItemIdx.values[ tr_start_offset[iters]]
                feed_dict = {model.x: in_idx, model.y: out_idx,
                             model.curr: tr_start_offset[iters], model.end: tr_end_offset[iters]}
                for i in range(args.rnn_layers):
                    feed_dict[model.state[i]] = gru_state[i]
                fetches = [model.sampled_cost, model.final_state, model.mask, model.mask_num, model.global_step, model.train_op]
                cost, state, mask, mask_num, global_step, _ = sess.run(fetches, feed_dict)
                if mask_num != 0:
                    iters[mask] = max_iter + 1 + np.arange(mask_num)
                    max_iter = iters.max()
                    for i in range(args.rnn_layers):
                        gru_state[i][mask] = 0
                if max_iter >= max_sess:
                    break

                if global_step % args.period == 0:
                    loss_board = tf.Summary(value=[tf.Summary.Value(
                        tag="train/loss", simple_value=cost),])
                    writer.add_summary(loss_board, global_step)
                    writer.flush()

                if global_step % args.chkpnt_period == 0:
                    sampling = np.zeros([len(va_start_offset)], dtype=np.int32)
                    sampling[:args.valid_num] = 1
                    sampling = sampling.astype(np.bool)
                    np.random.shuffle(sampling)
                    smp_start_offset = np.array(va_start_offset[sampling])
                    smp_end_offset = np.array(va_end_offset[sampling])
                    smp_chk_offset = np.array(va_chk_offset[sampling])
                    sess.run(tf.assign(model.dropout, tf.constant(1, dtype=tf.float32)))
                    recall, mrr, v_cost = test_model(args, data, model, sess, smp_start_offset, smp_end_offset, smp_chk_offset,
                                             is_valid=True)
                    sess.run(tf.assign(model.dropout, tf.constant(args.dropout, dtype=tf.float32)))
                    v_loss_board = tf.Summary(value=[tf.Summary.Value(
                        tag='valid/loss', simple_value=v_cost), ])
                    v_recall_board = tf.Summary(value=[tf.Summary.Value(
                        tag='valid/recall', simple_value=recall), ])
                    v_mrr_board = tf.Summary(value=[tf.Summary.Value(
                        tag='valid/mrr', simple_value=mrr), ])

                    writer.add_summary(v_loss_board, global_step)
                    writer.add_summary(v_recall_board, global_step)
                    writer.add_summary(v_mrr_board, global_step)



            print("#", epoch, "Epoch End")
    return

def test_model(args, data, model, sess, start_offset, end_offset, chk_offset, is_valid=False):
    iters = np.arange(args.batch_size)
    max_iter = iters.max()
    gru_state = [np.zeros([args.batch_size, args.hidden_size], dtype=np.float32) for _ in range(args.rnn_layers)]
    
    recall = 0
    mrr = 0
    evaluation_point_count = 0
    max_sess = len(start_offset) - 1
    total_cost = []

    while True:
        in_idx = data.ItemIdx.values[start_offset[iters]]
        start_offset[iters] += 1
        out_idx = data.ItemIdx.values[start_offset[iters]]
        feed_dict = {model.x : in_idx, model.y : out_idx,
                     model.curr : start_offset[iters], model.end : end_offset[iters],
                     model.chk  : chk_offset[iters]}

        for i in range(args.rnn_layers):
            feed_dict[model.state[i]] = gru_state[i]

        if is_valid:
            fetches = [model.cost, model.yhat, model.final_state, model.mask, model.mask_num, model.chk_mask]
            cost, yhat, state, mask, mask_num, chk_mask = sess.run(fetches, feed_dict)
            total_cost.append(cost)
        else:
            fetches = [model.yhat, model.final_state, model.mask, model.mask_num, model.chk_mask]
            yhat, state, mask, mask_num, chk_mask = sess.run(fetches, feed_dict)

        if chk_mask.sum() != 0:
            preds = yhat[chk_mask]
            label_probs = np.diag(preds.T[out_idx[chk_mask]]).reshape(-1, 1)
            ranks = (preds > label_probs).sum(axis=1) + 1
            rank_ok = ranks < 20
            recall += rank_ok.sum()
            mrr += (1.0/ranks[rank_ok]).sum()
            evaluation_point_count += len(ranks)
        if mask_num != 0:
            iters[mask] = max_iter + 1 + np.arange(mask_num)
            max_iter = iters.max()
            for i in range(args.rnn_layers):
                gru_state[i][mask] = 0
        if max_iter >= max_sess:# or :
            break
    return recall/evaluation_point_count, mrr/evaluation_point_count, sum(total_cost) / len(total_cost)

def test(args):
    print(' * Test', args.dataset)
    return

def get_offset_data(args, offset_sessions):
    offset_sessions = np.array(offset_sessions)
    offset_len = offset_sessions[1:] - offset_sessions[:-1]

    tr_start_offset = np.array(offset_sessions[:-1])
    tr_end_offset = np.array(tr_start_offset + (offset_len * args.train_ratio).astype(np.int32))

    va_start_offset = np.array(tr_start_offset)
    va_end_offset = np.array(va_start_offset + (offset_len * (args.train_ratio+args.valid_ratio)).astype(np.int32))
    va_chk_offset = np.array(tr_end_offset)

    te_start_offset = np.array(va_start_offset)
    te_end_offset = np.array(offset_sessions[1:])
    te_chk_offset = np.array(va_end_offset)

    result = { 'tr' : (tr_start_offset, tr_end_offset - 1),
               'va' : (va_start_offset, va_end_offset - 1, va_chk_offset - 1),
               'te' : (te_start_offset, te_end_offset - 1, te_chk_offset - 1) }

    return result

def make_ItemIdx(data):
    itemids = data['ItemId'].unique()
    n_items = len(itemids)
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids)
    data = pd.merge(data, pd.DataFrame({'ItemId': itemids, 'ItemIdx': itemidmap[itemids].values}),
                    on='ItemId', how='inner')
    return data, n_items
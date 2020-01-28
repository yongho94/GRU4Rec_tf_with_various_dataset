import pandas as pd
from model import Model
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import time
from pprint import pprint
from collections import defaultdict
import random

def train(args):
    print(' * Train', args.dataset)

    print(' * Load Dataset......')
    data_path = os.path.join(args.data_dir, args.dataset)
    user_count, item_count, user_ratings = load_data(data_path)
    train_set = list()
    valid_set = list()
    test_set = list()
    for idx, u in tqdm(enumerate(user_ratings), desc='Make uij pair'):
        item_list = list(user_ratings[u])
        random.shuffle(item_list)
        tr_idx = int(len(item_list) * args.train_ratio)
        va_idx = int(len(item_list) * (args.train_ratio + args.valid_ratio))

        tr_item = item_list[:tr_idx]
        va_item = item_list[tr_idx:va_idx]
        te_item = item_list[va_idx:]

        train_set += make_data_point(u, item_list, tr_item, item_count)
        valid_set += make_data_point(u, item_list, va_item, item_count)
        test_set += make_data_point(u, item_list, te_item, item_count)

    print(' * Loading Model......')
    model = Model(args, user_count, item_count, is_train=True)

    print(' * Start training...!')
    sess_config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(args.log_dir)
        
        for epoch in range(args.n_epochs):  # Epoch
            print("#", epoch, "Epoch Start....")
            random.shuffle(train_set)
            random.shuffle(valid_set)
            random.shuffle(test_set)
            _batch_bprloss = 0
            batch_idxs = [i for i in range(int(len(train_set) / args.batch_size))]
            for i in tqdm(batch_idxs, "Step length"):
                idx = i * args.batch_size
                b_data = np.array(train_set[idx:idx + args.batch_size])
                global_step, _bprloss, auc, _ = sess.run([model.global_step, model.bprloss, model.auc, model.train_op],
                                          feed_dict={model.u: b_data[:, 0], model.i: b_data[:, 1], model.j: b_data[:, 2]})
                _batch_bprloss += _bprloss

                if global_step % args.period == 0:
                    loss_board = tf.Summary(value=[tf.Summary.Value(
                        tag="train/loss", simple_value=_bprloss),])
                    auc_board = tf.Summary(value=[tf.Summary.Value(
                        tag="train/auc", simple_value=auc),])
                    writer.add_summary(loss_board, global_step)
                    writer.add_summary(auc_board, global_step)
                    writer.flush()

                if global_step % args.chkpnt_period == 0:
                    v_loss, v_auc = test_model(args, valid_set, model, sess, is_valid=True)
                    v_loss_board = tf.Summary(value=[tf.Summary.Value(
                        tag="valid/loss", simple_value=v_loss),])
                    v_auc_board = tf.Summary(value=[tf.Summary.Value(
                        tag="valid/auc", simple_value=v_auc),])

                    writer.add_summary(v_loss_board, global_step)
                    writer.add_summary(v_auc_board, global_step)
                    writer.flush()

            print("#", epoch, "Epoch End")
    return

def test_model(args, data, model, sess, is_valid=False):

    batch_idxs = [i for i in range(int(len(data) / args.batch_size))]
    loss_list = []
    auc_list = []
    for i in tqdm(batch_idxs):
        idx = i * args.batch_size
        b_data = np.array(data[idx:idx+args.batch_size])
        _bprloss, auc = sess.run([model.bprloss, model.auc], feed_dict={
            model.u:b_data[:,0], model.i:b_data[:, 1], model.j:b_data[:, 2]})
        print(_bprloss, auc)
        loss_list.append(_bprloss)
        auc_list.append(auc)
    return sum(loss_list)/len(loss_list), sum(auc_list)/len(auc_list)



def test(args):
    print(' * Test', args.dataset)
    return

def make_ItemIdx(data):
    itemids = data['ItemId'].unique()
    n_items = len(itemids)
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids)
    data = pd.merge(data, pd.DataFrame({'ItemId': itemids, 'ItemIdx': itemidmap[itemids].values}),
                    on='ItemId', how='inner')
    return data, n_items

def make_data_point(u, item_list, label_item, item_count):
    result = list()
    for i in label_item:
        while True:
            j = random.randint(0, item_count)
            if j not in item_list:
                result.append([u, i, j])
                break
    return result

def load_data(data_path):

    user_ratings = defaultdict(set)
    data = pd.read_csv(data_path, sep='\t', dtype={0: np.int64, 1: np.int64, 2: np.int64})
    data, n_items = make_ItemIdx(data)
    data = data[['SessionId', 'ItemIdx']]
    userId = list(data['SessionId'])
    itemId = list(data['ItemIdx'])
    for u, i in tqdm(zip(userId, itemId), desc='Load dataset'):
        user_ratings[int(u)].add(int(i))
    max_u_id = max(userId)
    max_i_id = max(itemId)

    return max_u_id, max_i_id, user_ratings
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from pprint import pprint
import utils
import numpy as np

class Model():
    def __init__(self, args, user_count, item_count, is_train=True):
        self.is_train = is_train
        self.global_step = tf.get_variable('global_step', shape=[],
                                           dtype=tf.int32, initializer=tf.constant_initializer(1), trainable=False)
        self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)


        ## Placeholder
        self.u = tf.placeholder(tf.int32, [args.batch_size])  # user
        self.i = tf.placeholder(tf.int32, [args.batch_size])  # positive item
        self.j = tf.placeholder(tf.int32, [args.batch_size])  # negative item


        # Model
        user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, args.hidden_size],
                                     initializer=tf.random_normal_initializer(0, 0.1))
        item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, args.hidden_size],
                                     initializer=tf.random_normal_initializer(0, 0.1))
        item_b = tf.get_variable("item_b", [item_count + 1, 1],
                                 initializer=tf.constant_initializer(0.0))

        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
        i_b = tf.nn.embedding_lookup(item_b, self.i)
        j_emb = tf.nn.embedding_lookup(item_emb_w, self.j)
        j_b = tf.nn.embedding_lookup(item_b, self.j)

        x = i_b - j_b + tf.reduce_sum(tf.matmul(u_emb, (i_emb - j_emb), transpose_b=True), 1, keep_dims=True)
        self.auc = tf.reduce_mean(tf.to_float(x > 0))

        l2_norm = tf.add_n([
            tf.reduce_sum(tf.matmul(u_emb, u_emb, transpose_b=True)),
            tf.reduce_sum(tf.matmul(i_emb, i_emb, transpose_b=True)),
            tf.reduce_sum(tf.matmul(j_emb, j_emb, transpose_b=True))
        ])
        regulation_rate = 0.0001
        self.bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        vars = tf.trainable_variables()
        grads = optimizer.compute_gradients(self.bprloss, vars)
        self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)

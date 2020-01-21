import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from pprint import pprint
import utils
import numpy as np

class Model():
    def __init__(self, args, n_items, is_train=True):
        self.is_train = is_train
        self.global_step = tf.get_variable('global_step', shape=[],
                                           dtype=tf.int32, initializer=tf.constant_initializer(1), trainable=False)
        self.dropout = tf.get_variable('dropout', shape=[],
                                       dtype=tf.float32, initializer=tf.constant_initializer(args.dropout),
                                       trainable=False)
        self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)

        self.gru_act, self.final_activation, self.loss_function = utils.initialize_func(args)

        ## Placeholder
        self.x = tf.placeholder(tf.int32, [args.batch_size])  # input
        self.y = tf.placeholder(tf.int32, [args.batch_size])  # output
        self.state = [tf.placeholder(tf.float32, [args.batch_size, args.hidden_size]) for _ in range(args.rnn_layers)]
        print(self.state)
        
        ## For Masking
        self.curr = tf.placeholder(tf.int32, [args.batch_size])
        self.end = tf.placeholder(tf.int32, [args.batch_size])
        self.chk = tf.placeholder(tf.int32, [args.batch_size])
        self.mask = tf.math.equal(self.curr, self.end)
        self.mask_num = tf.count_nonzero(self.mask)
        self.chk_mask = tf.math.greater(self.curr, self.chk)
        
        ## Initializing Parameters
        sigma = args.sigma if args.sigma != 0 else np.sqrt(6.0 / args.hidden_size - 1)
        if args.init_as_normal:
            initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
        else:
            initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)

        embedding = tf.get_variable('embeddeing', [n_items, args.hidden_size],
                                    initializer=initializer)
        softmax_W = tf.get_variable('softmax_W', [n_items, args.hidden_size],
                                    initializer=initializer)
        softmax_b = tf.get_variable('softmax_b', [n_items],
                                    initializer=tf.constant_initializer(0.0))

        # what activation function should be used?

        cell = rnn_cell.GRUCell(args.hidden_size, activation=self.gru_act)
        drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
        stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * args.rnn_layers)

        inputs = tf.nn.embedding_lookup(embedding, self.x)
        output, state = stacked_cell(inputs, tuple(self.state))
        self.final_state = state


        if self.is_train == True:
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.y)
            sampled_logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.sampled_yhat = self.final_activation(sampled_logits)
            self.sampled_cost = self.loss_function(self.sampled_yhat)#tf.reduce_mean(-tf.log(tf.diag_part(sampled_logits)+1e-24))

            self.lr = tf.maximum(1e-5,
                    tf.train.exponential_decay(args.lr, self.global_step, args.decay_steps,
                                                            args.decay, staircase=True))
            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            pprint(tvars)
            grads = optimizer.compute_gradients(self.sampled_cost, tvars)

            if args.grad_cap > 0:
                capped_grads = [(tf.clip_by_norm(grad, self.grad_cap), var)
                              for grad, var in grads]
            else:
                capped_grads = grads
            self.train_op = optimizer.apply_gradients(capped_grads,
                                                      global_step=self.global_step)

        logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
        self.yhat = self.final_activation(logits)
        self.cost = self.loss_function(self.yhat)
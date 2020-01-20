import tensorflow as tf

def initialize_func(args):

    if args.gru_act == 'tanh':
        gru_act = tanh
    elif args.hidden_act == 'relu':
        gru_act = relu
    else:
        raise NotImplementedError

    if args.loss == 'cross-entropy':
        if args.final_act == 'tanh':
            final_act = softmaxth
        else:
            final_act = softmax
        loss_function = cross_entropy

    elif args.loss == 'bpr':
        if args.final_act == 'linear':
            final_act = linear
        elif args.final_act == 'relu':
            final_act = relu
        else:
            final_act = tanh
        loss_function = bpr

    elif args.loss == 'top1':
        if args.final_act == 'linear':
            final_act = linear
        elif args.final_act == 'relu':
            final_act = relu
        else:
            final_act = tanh
        loss_function = get_top1(args)
    else:
        raise NotImplementedError

    return gru_act, final_act, loss_function

def linear(X):
    return X
def tanh(X):
    return tf.nn.tanh(X)
def softmax(X):
    return tf.nn.softmax(X)
def softmaxth(X):
    return tf.nn.softmax(tf.tanh(X))
def relu(X):
    return tf.nn.relu(X)
def sigmoid(X):
    return tf.nn.sigmoid(X)
def cross_entropy(yhat):
    return tf.reduce_mean(-tf.log(tf.linalg.diag_part(yhat)+1e-24))
def bpr(yhat):
    yhatT = tf.transpose(yhat)
    return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))

def get_top1(args):
    def top1(yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.linalg.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
        term2 = tf.nn.sigmoid(tf.linalg.diag_part(yhat)**2) / args.batch_size
        return tf.reduce_mean(term1 - term2)
    return top1
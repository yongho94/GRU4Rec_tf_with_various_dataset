import argparse
from main import *

def get_config():
    
    ############ only for jupyter , erase it later
    dataset = 'prcd_ratings_Books.txt'
    mode = 'train'
    ############
    parser = argparse.ArgumentParser()

    parser.add_argument('--fname', type=str, default='v01')
    # Dataset
    parser.add_argument('--data_dir', type=str, default='./../dataset', help='raw dataset dir')
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset name')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--chkpnt', type=str, default='checkpoint')

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--valid_num', type=int, default=10000)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    parser.add_argument('--mode', type=str, default=mode)

    # Hyperparameter
    parser.add_argument('--init_as_normal', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.96)
    parser.add_argument('--decay_steps', type=int, default=1e4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--grad_cap', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--gru_act', type=str, default='tanh')
    parser.add_argument('--final_act', type=str, default='softmax')
    parser.add_argument('--loss', type=str, default='cross-entropy')
    parser.add_argument('--reset_after_session', type=bool, default=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_config()
    
    assert args.dataset != None
    assert args.mode in ['train', 'test']
    args.log_dir = os.path.join(args.log_dir, args.fname)
    args.chkpnt = os.path.join(args.chkpnt, args.fname)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.chkpnt):
        os.mkdir(args.chkpnt)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print('Wrong dataset')


import argparse

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs for train [default: 256]')
parser.add_argument('--log_interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('--test_interval', type=int, default=500, help='how many steps to wait before testing [default: 100]')
parser.add_argument('--save_dir', type=str, default=None, help='where to save the snapshot')
parser.add_argument('--training_model', type=str, default=None, help='to continue training')

# data
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training [default: 64]')

# model
parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('--embedding_dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('--channel_out', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('--mode', action='store_true', default='CNN-static', help='fix the embedding')

# option
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('--predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('--test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

def get_config():
    return args
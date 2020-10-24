import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id')
parser.add_argument('--train_log_path')
parser.add_argument('--num_epoch')

parser.add_argument('--in_channels')
parser.add_argument('--out_channels')
parser.add_argument('--filters')
parser.add_argument('--norm', default='inorm')

parser.add_argument('--wgt_cycle')
parser.add_argument('--wgt_ident')
parser.add_argument('--lr')
parser.add_argument('--batch_size')
parser.add_argument('--num_works')

import argparse

from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', default=None)
parser.add_argument('--train_log_path', default='.')
parser.add_argument('--num_epoch', default=100, type=int)

parser.add_argument('--in_channels', default=3, type=int)
parser.add_argument('--out_channels', default=3, type=int)
parser.add_argument('--filters', default=64)
parser.add_argument('--norm', default='inorm')

parser.add_argument('--wgt_cycle', default=10)
parser.add_argument('--wgt_ident', default=0.5)
parser.add_argument('--lr', default=0.0002)
parser.add_argument('--batch_size', default=1)
parser.add_argument('--num_works', default=0)

opt = parser.parse_args()


def main():
    trainer = Trainer(opt)
    trainer.train()


if __name__ == '__main__':
    main()

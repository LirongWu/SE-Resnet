#-*- coding: UTF-8 -*-
import argparse
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFXP')

    parser.add_argument('--dataset_path', type=str, default='cifar-10-batches-py', help='Dataset path')

    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=100, help='Number of training epoch')
    parser.add_argument('--n_gpu', type=str, default='3,4', help='Number of GPUs')
    parser.add_argument('--ratio', type=int, default=4, help='Ratio for SENet/GCNet')
    parser.add_argument('--mode', type=str, default='SENet', help='Mode: SENet, NLNet, SNLNet and GCNet')

    params = parser.parse_args()

    model = Model(params)
    model.train()

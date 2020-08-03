import argparse
import sys

from networks import NETWORKS

def get_args(dataset):
    if dataset == 'cifar10':
        return cifar10_args()
    elif dataset == 'imagenet':
        return imagenet_args()
    elif dataset == 'furniture':
        return furniture_args()
    else:
        raise('Unknown dataset')

def cifar10_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--image_dir', '-I', default='', type=str, metavar='ID', help='location of image data')
    parser.add_argument('--network_dir', '-N', default='', type=str, metavar='ND', help='location of network data')
    parser.add_argument('--data_dir', '-D', default='', type=str, metavar='DD', help='location of parsed data')
    parser.add_argument('--results_dir', '-R', default='', type=str, metavar='RD', help='location of results')
    parser.add_argument('--network', '-n', default='resnet18_modified', type=str, metavar='N', choices=NETWORKS, help='network architecture')
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='BS', help='Batch size')
    parser.add_argument('--learning_rate', '-l', default=0.1, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', '-w', default=0.0001, type=float, metavar='WD', help='weight decay')
    parser.add_argument('--momentum', '-m', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--num_epochs', '-e', default=80, type=int, metavar='NE', help='number of epochs to train for')
    parser.add_argument('--gpu', '-g', default='cuda:0', type=str, metavar='G', choices=['cuda:0', 'cuda:1'], help='gpu id (e.g. \'cuda:0\'')
    args = parser.parse_args()
    args = vars(args)
    return args

def furniture_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-Furniture Training')
    parser.add_argument('--image_dir', '-I', default='', type=str, metavar='ID', help='location of image data')
    parser.add_argument('--network_dir', '-N', default='', type=str, metavar='ND', help='location of network data')
    parser.add_argument('--data_dir', '-D', default='', type=str, metavar='DD', help='location of parsed data')
    parser.add_argument('--results_dir', '-R', default='', type=str, metavar='RD', help='location of results')
    parser.add_argument('--network', '-n', default='resnet34', type=str, metavar='N', choices=NETWORKS, help='network architecture')
    parser.add_argument('--batch_size', '-b', default=32, type=int, metavar='BS', help='Batch size')
    parser.add_argument('--learning_rate', '-l', default=0.01, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', '-w', default=0.0001, type=float, metavar='WD', help='weight decay')
    parser.add_argument('--momentum', '-m', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--num_epochs', '-e', default=200, type=int, metavar='NE', help='number of epochs to train for')
    parser.add_argument('--gpu', '-g', default='cuda:0', type=str, metavar='G', choices=['cuda:0', 'cuda:1'], help='gpu id (e.g. \'cuda:0\'')
    args = parser.parse_args()
    args = vars(args)
    return args

def imagenet_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--image_dir', '-I', default='', type=str, metavar='ID', help='location of image data')
    parser.add_argument('--network_dir', '-N', default='', type=str, metavar='ND', help='location of network data')
    parser.add_argument('--data_dir', '-D', default='', type=str, metavar='DD', help='location of parsed data')
    parser.add_argument('--results_dir', '-R', default='', type=str, metavar='RD', help='location of results')
    parser.add_argument('--network', '-n', default='resnet34', type=str, metavar='N', choices=NETWORKS, help='network architecture')
    parser.add_argument('--batch_size', '-b', default=32, type=int, metavar='BS', help='Batch size')
    parser.add_argument('--learning_rate', '-l', default=0.01, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', '-w', default=0.0001, type=float, metavar='WD', help='weight decay')
    parser.add_argument('--momentum', '-m', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--num_epochs', '-e', default=400, type=int, metavar='NE', help='number of epochs to train for')
    parser.add_argument('--gpu', '-g', default='cuda:0', type=str, metavar='G', choices=['cuda:0', 'cuda:1'], help='gpu id (e.g. \'cuda:0\'')
    args = parser.parse_args()
    args = vars(args)
    return args
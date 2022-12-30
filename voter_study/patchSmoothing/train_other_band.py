from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import resnet_imgnt as resnet

from prog_bar import progress_bar
import utils_band as utils

import sys
sys.path.append('../../')
from data_loaders import *
from pathlib import Path
from datetime import datetime
import time

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
        
def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
        
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--band_size', default=20, type=int, help='band size')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--regularization', default=0, type=float, help='weight decay')
parser.add_argument('--resume', default=None, help='resume from checkpoint', type=str)
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--end_epoch', default=60, type=int, help='end_epoch')
#parser.add_argument('--trainpath', default='imagenet-train/train', type=str, help='Path to ImageNet training set')
#parser.add_argument('--valpath', default='imagenet-val/val', type=str, help='Path to ImageNet validation set')
parser.add_argument('--trainpath', default='imagenet-train/train', type=str, help='Path to ImageNet training set')
parser.add_argument('--valpath', default='imagenet-val/val', type=str, help='Path to ImageNet validation set')
parser.add_argument('--checkpoint_dir', default="checkpoints", type=str, help="The path for checkpoint")
parser.add_argument("--dataset", type=str, default='ImageNet', 
                    choices=["GTSRB", "ImageNet", "CIFAR10", "CelebA", "FER2013", "FOOD101"], 
                    help="dataset for fine-tunning/evaluation")
parser.add_argument('--data-dir', metavar='DIR', help='path to dataset')
parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus to use")
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = args.checkpoint_dir
checkpoint_file = f'./{checkpoint_dir}/{args.dataset}_one_band_lr_{args.lr}_regularization_{args.regularization}_model_{args.model}__band_{args.band_size}_epoch_{args.end_epoch}.pth'

print("==> Checkpoint directory", checkpoint_dir)
print("==> Saving to", checkpoint_file)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
dataset_dir = args.dataset
trainloader = eval("{}DataLoader".format(args.dataset))(
                data_dir=os.path.join(args.data_dir, dataset_dir),
                image_size=224,
                mask_size=-1,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split='train',
                model_type="resnet50") # Model type is used for normalization config
nomtestloader = eval("{}DataLoader".format(args.dataset))(
                data_dir=os.path.join(args.data_dir, dataset_dir),
                image_size=224,
                mask_size=-1,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split='validation',
                model_type="resnet50")

# Model
print('==> Building model..')
if (args.model == 'resnet50'):
    net = resnet.resnet50(pretrained=True)
elif (args.model == 'resnet18'):
    net = resnet.resnet18(pretrained=True)

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, args.num_classes)
    
if args.n_gpu is not None:
    device, device_ids = setup_device(args.n_gpu)
print("Using device:{}".format(device_ids))
net = net.to(device)

if device != 'cpu':
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    #resume_file = '{}/{}'.format(checkpoint_dir, args.resume)
    resume_file = args.resume
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    checkpoint_file = './{}/{}_one_band_lr_{}_regularization_{}_band_{}_epoch_{}_resume_{}.pth'.format(checkpoint_dir, args.dataset, args.lr,args.regularization, args.band_size, args.end_epoch, args.resume)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.regularization)

# Training
def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
            len(trainloader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
    print('\nEpoch: %d' % epoch)
    net.train()
    end = time.time()
    train_loss = 0
    correct = 0
    total_epsilon = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        outputs = net(utils.random_mask_batch_one_sample(inputs, args.band_size, reuse_noise=True))
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        loss.backward()
        optimizer.step()
        total += targets.size(0)

        train_loss += loss.item()
        
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
def test_nominal(epoch, best_acc):
    print('\nEpoch: %d' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(nomtestloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            #breakpoint()
            outputs = net(utils.random_mask_batch_one_sample(inputs, args.band_size, reuse_noise=True))
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
            total += targets.size(0)
            #progress_bar(batch_idx, len(nomtestloader), 'Nominal Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    print("[EVAL] epoch:{}, acc:{}".format(epoch, acc))
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, checkpoint_file)
        #torch.save(state, "checkpoints/imagenet_" + str(epoch)+".pth")



if __name__ == '__main__':
    best_acc = 0
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Begin Experiment: timestamp:{}".format(dt_string))
    ensure_dir(args.checkpoint_dir)
    for epoch in range(start_epoch,  args.end_epoch+1):
        train(epoch)
        test_nominal(epoch, best_acc)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End Experiment: timestamp:{}".format(dt_string))

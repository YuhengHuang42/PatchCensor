from __future__ import print_function

import sys


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse

from prog_bar import progress_bar
import utils_band as utils

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--band_size', default=2, type=int, help='band size')
parser.add_argument('--regularization', default=0.0005, type=float, help='weight decay')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--end_epoch', default=400, type=int, help='end_epoch')
parser.add_argument('--threeC', action='store_true', help='whether to train on three channel dataset')
parser.add_argument('--trainPath', default=None, type=str, help="the training set file path")
parser.add_argument('--testPath', default=None, type=str, help="the test set file path")
parser.add_argument('--description', default='', type=str, help="description for the model. Used in checkpoint file name")

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = '../checkpoints'
if not os.path.exists('../checkpoints'):
    os.makedirs('../checkpoints')
if args.threeC:
    cmode = "threeC"
    channelN = 6
else:
    cmode = "singleC"
    channelN = 2
checkpoint_file = f'./{checkpoint_dir}/mnist_one_band_lr_{args.lr}_epoch_{args.end_epoch}_{cmode}{args.description}.pth'

print("==> Checkpoint directory", checkpoint_dir)
print("==> Saving to", checkpoint_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

if args.threeC == False:
    trainset = torchvision.datasets.MNIST(root='../../split_verified/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='../../split_verified/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
#nomtestloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

else:
    trainset = torch.load(args.trainPath)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torch.load(args.testPath)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
net = nn.Sequential(
        nn.Conv2d(channelN, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(128*7*7,500),
        nn.ReLU(),
        nn.Linear(500,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    resume_file = '{}/{}'.format(checkpoint_dir, args.resume)
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    checkpoint_file = './{}/mnist_one_band_lr_{}_regularization_{}_band_{}_epoch_{}_resume_{}.pth'.format(checkpoint_dir, args.lr,args.regularization, args.band_size,'{}', args.resume)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.regularization)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.end_epoch*200//400], gamma=0.1)

best_acc = 0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total_epsilon = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(utils.random_mask_batch_one_sample(inputs, args.band_size, reuse_noise=True))
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total += targets.size(0)

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
def test(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            #breakpoint()
            outputs = net(utils.random_mask_batch_one_sample(inputs, args.band_size, reuse_noise=True))
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
            total += targets.size(0)
            progress_bar(batch_idx, len(testloader), 'Nominal Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100. * correct / total
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
    return acc



for epoch in range(start_epoch,  args.end_epoch+1):
    train(epoch)
    acc = test(epoch)
    print("test acc: {}".format(acc))

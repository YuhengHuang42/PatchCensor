##############################################################################
# Adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
##############################################################################

'''Train other datasets with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
import sys

sys.path.append('../')
sys.path.append('../../')
import nets.bagnet
import nets.resnet
from utility import apply_transform_Tensordataset

import PIL

#from utils.progress_bar import progress_bar

import numpy as np  
import joblib

import random

parser = argparse.ArgumentParser(description='PyTorch traffic sign Training')
parser.add_argument('--trainPath', type=str, help="the training set file path")
parser.add_argument('--testPath', type=str, help="the test set file path")
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, help="The name for model save of the path for pre-trained model")
parser.add_argument("--clip",default=-1,type=int)
parser.add_argument('--classNum', default=43, type=int)
parser.add_argument('--advTrain', action='store_true', help='whether to use Robust training')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

arch = "bagnet"
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    #transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torch.load(args.trainPath)
trainset = apply_transform_Tensordataset(trainset, transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

testset = torch.load(args.testPath)
testset = apply_transform_Tensordataset(testset, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)


if args.clip > 0:
	clip_range = [0,args.clip]
else:
	clip_range = None

# Model
print('==> Building model..')

if not os.path.exists('../checkpoints'):
    os.makedirs('../checkpoints')

pth_path = os.path.join('../checkpoints', args.checkpoint)

if not os.path.exists('./log'):
    os.makedirs('./log')
logs = open(os.path.join(
    'log', 'dataset_{}_arch{}_lr{}_epoch{}_'.format(args.trainPath.split("/")[-1].split("_")[0], arch, args.lr, 20)
    ), 'w')


if args.advTrain == True:
    aggregation = 'adv'
else:
    aggregation = 'mean'

net = nets.bagnet.bagnet17(pretrained=True, clip_range=clip_range, aggregation=aggregation) #aggregation = 'adv' for provable adversarial training

#net = nets.resnet.resnet50(pretrained=True)

#for param in net.parameters():
#    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, args.classNum)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('../checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(pth_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training
def train(epoch, advTrain=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if advTrain == True:
            outputs = net(inputs, targets)
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % (len(trainloader) // 10) == 0:
            print("batch:{}, total batch:{}, loss:{:.3f}, acc:{:.3f}".format(batch_idx, len(trainloader), 
                    train_loss/(batch_idx+1), 100.*correct/total))
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx_list=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    #joblib.dump(idx_list,'masked_contour_correct_idx.z')
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, pth_path)
        best_acc = acc
    return acc

# python train_cifar.py --lr 0.01 
# python train_cifar.py --resume --lr 0.001 
print("begin training")
start_time = time.time()
for epoch in range(start_epoch, start_epoch+20):
    train(epoch, args.advTrain)
    acc = test(epoch)
    scheduler.step()
    logs.writelines("model:{}, epoch:{}, test_acc:{}, time:{}\n".format(arch, epoch, 
                                                                           acc, round(time.time()-start_time, 2)))
    logs.flush()
    start_time = time.time()

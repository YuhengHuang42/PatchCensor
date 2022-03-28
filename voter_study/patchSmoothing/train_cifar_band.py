from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pytorch_cifar.models.resnet as resnet

import torchvision
import torchvision.transforms as transforms
import os
import argparse

from prog_bar import progress_bar
import utils_band as utils
import PIL

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--band_size', default=4, type=int, help='band size')
parser.add_argument('--regularization', default=0.0005, type=float, help='weight decay')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--end_epoch', default=199, type=int, help='end_epoch')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--resize', default='32', help='the H/W after the resize', type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.resize <= 32:
    assert (32 - args.resize) % 2 == 0
    padding_size = (32 - args.resize) // 2
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize((args.resize, args.resize), PIL.Image.BICUBIC),
        transforms.Pad(padding_size, padding_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.Resize((args.resize, args.resize), PIL.Image.BICUBIC),
        transforms.Pad(padding_size, padding_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    sampling_mode = "downscale"
else:
    transform_train = transforms.Compose([
        transforms.Resize((args.resize, args.resize), PIL.Image.BICUBIC),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.resize, args.resize), PIL.Image.BICUBIC),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    sampling_mode = "upscale"


checkpoint_dir = 'checkpoints'
if not os.path.exists('../checkpoints'):
    os.makedirs('../checkpoints')
    
checkpoint_file = f'../{checkpoint_dir}/cifar_lr_{args.lr}_model_resize_{args.resize}_{args.model}_band_{args.band_size}_{sampling_mode}_epoch_{{}}.pth'

print("==> Checkpoint directory", checkpoint_dir)
print("==> Saving to", checkpoint_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

trainset = torchvision.datasets.CIFAR10(root='../../dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)
nomtestloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

if (args.model == 'resnet50'):
    net = resnet.ResNet50()
elif (args.model == 'resnet18'):
    net = resnet.ResNet18()

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
    checkpoint_file = './{}/cifar_one_band_lr_{}_regularization_{}_model_{}_band_{}_epoch_{}_resume_{}.pth'.format(checkpoint_dir, args.lr,args.regularization, args.model, args.band_size,'{}', args.resume)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.regularization)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.end_epoch*150//350,args.end_epoch*250//350], gamma=0.1)

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
 
best_acc = 0
def test_nominal(epoch):
    global best_acc
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
            progress_bar(batch_idx, len(nomtestloader), 'Nominal Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if best_acc < acc:
        best_acc = acc
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, checkpoint_file.format(args.end_epoch))



for epoch in range(start_epoch,  args.end_epoch+1):
    train(epoch)
    test_nominal(epoch)
    scheduler.step()

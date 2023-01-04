import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import resnet_imgnt as resnet
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import utils_band as utils
from prog_bar import progress_bar

import sys
sys.path.append('../../')
sys.path.append('../')
from data_loaders import *
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch Other dataset Certification')


parser.add_argument('--band_size', default=20, type=int, help='size of each smoothing band')
parser.add_argument('--size_to_certify', default=32, type=int, help='size_to_certify')
parser.add_argument("--model_dir", default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--checkpoint', help='checkpoint')
parser.add_argument('--threshhold', default=0.2, type=float, help='threshold for smoothing abstain')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument("--dataset", type=str, default='ImageNet', 
                    choices=["GTSRB", "ImageNet", "CIFAR10", "CelebA", "FER2013", "FOOD101"], 
                    help="dataset for fine-tunning/evaluation")
parser.add_argument('--data-dir', metavar='DIR', help='path to dataset')
parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
parser.add_argument("--device", type=str, default="cpu", help="specify which device to run")
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--num-workers", type=int, default=4, help="number of workers")

args = parser.parse_args()

device = args.device

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Begin Experiment: timestamp:{}".format(dt_string))

dataset_dir = args.dataset
testloader = eval("{}DataLoader".format(args.dataset))(
				data_dir=os.path.join(args.data_dir, dataset_dir),
				image_size=224,
				mask_size=-1,
				batch_size=args.batch_size,
				num_workers=args.num_workers,
				split='test',
				model_type="resnet")


# Model
print('==> Building model..')
checkpoint_dir = args.model_dir
if (args.model == 'resnet50'):
    net = resnet.resnet50()
elif (args.model == 'resnet18'):
    net = resnet.resnet18()

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, args.num_classes)

net = net.to(device)
if device != 'cpu':
    device_id = int(device.split(":")[-1])
    net = torch.nn.DataParallel(net, device_ids=[device_id])
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
#assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.checkpoint)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])
net.eval()


def test():
    global best_acc
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            predictions,  certyn = utils.predict_and_certify(inputs, net,args.band_size, args.size_to_certify, args.num_classes, device=device, threshold=args.threshhold)

            correct += (predictions.eq(targets)).sum().item()
            cert_correct += (predictions.eq(targets) & certyn).sum().item()
            cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()


            #progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d) Cert: %.3f%% (%d/%d)'  %  ((100.*correct)/total, correct, total, (100.*cert_correct)/total, cert_correct, total))
    print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(args.threshhold))
    print('Certifying For Patch ' +str(args.size_to_certify) + '*'+str(args.size_to_certify))
    print('Total images: ' + str(total))
    print('Correct: ' + str(correct) + ' (' + str((100.*correct)/total)+'%)')
    print('Certified Correct class: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
    print('Certified Wrong class: ' + str(cert_incorrect) + ' (' + str((100.*cert_incorrect)/total)+'%)')



test()


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Begin Experiment: timestamp:{}".format(dt_string))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torchvision import datasets, transforms

import nets.bagnet
import nets.resnet
from utils.defense_utils import *

import os 
import joblib
import argparse
from tqdm import tqdm
import numpy as np 
from scipy.special import softmax
from math import ceil
import PIL
from datetime import datetime

import sys
sys.path.append('../../')
sys.path.append('../')
from data_loaders import *

parser = argparse.ArgumentParser()

# python3 mask_bn_new.py --model bagnet17 --dataset imagenet --patch_size 32 --aggr adv --m 
#
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument("--testPath", type=str, help="patch to testDataset")
parser.add_argument("--model",default='bagnet17',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='none',type=str,help="aggregation methods. set to none for local feature")
parser.add_argument("--skip",default=1,type=int,help="number of example to skip")
parser.add_argument("--thres",default=0.0,type=float,help="detection threshold for robust masking")
parser.add_argument("--patch_size",default=31,type=int,help="size of the adversarial patch")
parser.add_argument("--m",action='store_true',help="use robust masking")
parser.add_argument("--cbn",action='store_true',help="use cbn")
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
parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
args = parser.parse_args()

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Begin Experiment: timestamp:{}".format(dt_string))
    
MODEL_DIR = os.path.join('.',args.model_dir)
#DATA_DIR = os.path.join(args.data_dir)

print("====Evaluating {} on patch size:{}====".format(args.dataset, args.patch_size))

#prepare data

# No transformation FIXME
#testset = torch.load(args.testPath)
#testset = apply_transform_Tensordataset(testset, transform_test)

#val_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)
dataset_dir = args.dataset
val_loader = eval("{}DataLoader".format(args.dataset))(
				data_dir=os.path.join(args.data_dir, dataset_dir),
				image_size=224,
				mask_size=-1,
				batch_size=args.batch_size,
				num_workers=args.num_workers,
				split='test',
				model_type="bagnet")
    
#build and initialize model
device = args.device

if args.clip > 0:
	clip_range = [0,args.clip]
else:
	clip_range = None

if 'bagnet17' in args.model:
    model = nets.bagnet.bagnet17(pretrained=True, clip_range=clip_range, aggregation=args.aggr)
    rf_size=17
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(pretrained=True, clip_range=clip_range, aggregation=args.aggr)
    rf_size=33
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(pretrained=True, clip_range=clip_range, aggregation=args.aggr)
    rf_size=9
rf_stride=8
window_size = ceil((args.patch_size + rf_size -1) / rf_stride)
print("window_size",window_size)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, args.num_classes)
device_id = int(device.split(":")[-1])
model = torch.nn.DataParallel(model, device_ids=[device_id])
cudnn.benchmark = True
print('restoring model from checkpoint...')
checkpoint = torch.load(os.path.join(MODEL_DIR, args.model+'.pth'))
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()

cudnn.benchmark = True

accuracy_list=[]
result_list=[]
clean_corr=0

if args.m:
    print("Using Robust masking")
elif args.cbn:
    print("Using CBN")
for data, labels in (val_loader):
	
	data = data.to(device)
	labels = labels.numpy()
	output_clean = model(data).detach().cpu().numpy() # logits
	#output_clean = softmax(output_clean,axis=-1) # confidence
	#output_clean = (output_clean > 0.2).astype(float) # predictions with confidence threshold
	
	#note: the provable analysis of robust masking is cpu-intensive and can take some time to finish
	#you can dump the local feature and do the provable analysis with another script so that GPU mempry is not always occupied
	for i in range(len(labels)):
		if args.m:#robust masking
			result = provable_masking(output_clean[i],labels[i],thres=args.thres,window_shape=[window_size,window_size])
			result_list.append(result)
			clean_pred = masking_defense(output_clean[i],thres=args.thres,window_shape=[window_size,window_size])
			clean_corr += clean_pred == labels[i]
		elif args.cbn:#cbn
			result = provable_clipping(output_clean[i],labels[i],window_shape=[window_size,window_size])
			result_list.append(result)
			clean_pred = clipping_defense(output_clean[i])
			clean_corr += clean_pred == labels[i]	
	acc_clean = np.mean(np.argmax(np.mean(output_clean,axis=(1,2)),axis=1) == labels)
	accuracy_list.append(acc_clean)


cases,cnt=np.unique(result_list,return_counts=True)

print("Provable robust accuracy:",cnt[-1]/len(result_list))
print("Clean accuracy with defense:",clean_corr/len(result_list))
print("Clean accuracy without defense:",np.mean(accuracy_list))
print("------------------------------")
print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):",cases)
print("Provable analysis breakdown",cnt/len(result_list))


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Begin Experiment: timestamp:{}".format(dt_string))
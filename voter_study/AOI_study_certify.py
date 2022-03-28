import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import seaborn as sn
from tqdm import tqdm
import random
import sys
import numpy as np
import argparse
import math

sys.path.append('./PatchGuard')
import nets.bagnet as bagnet
import nets.dsresnet_cifar as resnet
from utils.defense_utils import *

cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_cls = 10
dataset_path = "../dataset" # the path for dataset

def get_test_loader(model_arch, resize, batch_size=16, original_size=32):
    if "bagnet" in model_arch.lower():
        assert (original_size - resize) % 2 == 0
        assert resize <= original_size # downscale is not implemented for bagnet exp for now.
        padding_size = (original_size - resize) // 2
        transform_test = transforms.Compose([
            transforms.Resize((resize, resize), PIL.Image.BICUBIC),
            transforms.Pad(padding_size, padding_size),
            transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        if resize <= original_size:
            # downscale
            assert (original_size - resize) % 2 == 0
            padding_size = (original_size - resize) // 2
            transform_test = transforms.Compose([
                transforms.Resize((resize, resize), PIL.Image.BICUBIC),
                transforms.Pad(padding_size, padding_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            # upscale
            transform_test = transforms.Compose([
                transforms.Resize((resize, resize), PIL.Image.BICUBIC),
                transforms.CenterCrop(original_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader

def load_model(model_arch, path):
    if "bagnet" in model_arch.lower():
        net = bagnet.bagnet17(pretrained=True,clip_range=None,aggregation=None)
        net = net.to(device)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_cls)
        net = torch.nn.DataParallel(net)
        print("load bagnet")
    else:
        # band net
        net = resnet.ResNet18()
        net = net.to(device)
        net = torch.nn.DataParallel(net)
        print("load resnet")
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net

def certify_patchSmooth(test_loader, net, batch_size, band_size, patch_size):
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0
    B = batch_size
    predictions_all_list = torch.ones(B, num_cls)
    clean_corr_index = torch.ones(1)
    certified_index = torch.ones(1)
    wrong_index = torch.ones(1)
    labels = torch.ones(1)
    with torch.no_grad():
        for inputs, targets in (test_loader):
            #inputs = inputs * std + mean
            labels = torch.cat((labels, targets), dim=0)
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            predictions_verbose,  certyn = ds(inputs, net, band_size, 
                                            patch_size, num_cls, threshold = 0.2, verbose=True) # 0.2 is the parameters used in ds paper
            predictions_all_list = torch.cat((predictions_all_list, predictions_verbose.cpu()), dim=0)
            predinctionsnp = predictions_verbose.cpu().numpy()
            idxsort = np.argsort(-predinctionsnp,axis=1,kind='stable')
            predictions = torch.tensor(idxsort[:,0]).cuda()
            
            clean_corr_index = torch.cat((clean_corr_index, predictions.eq(targets).cpu()), dim=0)
            wrong_index = torch.cat((wrong_index, ~predictions.eq(targets).cpu()), dim=0)
            certified_index = torch.cat((certified_index, certyn.cpu()), dim=0)
            
            correct += (predictions.eq(targets)).sum().item()
            cert_correct += (predictions.eq(targets) & certyn).sum().item()
            cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()

    clean_corr_index = clean_corr_index[1:].type(torch.bool)
    wrong_index = wrong_index[1:].type(torch.bool)
    certified_index = certified_index[1:].type(torch.bool)
    labels = labels[1:]
    predictions_all_list = predictions_all_list[B:]
    correct_proportion = (100.*correct) / total
    cert_correct_proportion = (100.*cert_correct) / total
    cert_wrong_proportion = (100.*cert_incorrect) / total
    cert_proportion = (100.*(cert_correct + cert_incorrect)) / total
    result = {
        "correct_proportion": correct_proportion,
        "cert_correct_proportion": cert_correct_proportion,
        "cert_wrong_proportion": cert_wrong_proportion,
        "cert_proportion": cert_proportion,
        "clean_corr_index": clean_corr_index,
        "wrong_index": wrong_index,
        "certified_index": certified_index,
        "predictions_all_list": predictions_all_list
    }
    return result

def certify_patchGuard(test_loader, net, batch_size, band_size, patch_size):
    total = 0
    B = batch_size
    predictions_all_list = torch.ones(B, num_cls)
    clean_corr_index = torch.ones(1)
    certified_index = torch.ones(1)
    wrong_index = torch.ones(1)
    labels = torch.ones(1)
    result_list = []
    with torch.no_grad():
        for inputs, targets in (test_loader):
            #inputs = inputs * std + mean
            #original_images = torch.cat((original_images, inputs), dim=0)
            labels = torch.cat((labels, targets), dim=0)
            inputs = inputs.to(device)
            total += targets.size(0)
            result, clean_verbose_list = masking_ds(inputs, targets, net, band_size, patch_size, thres=0, verbose=True)
            predictions_all_list = torch.cat((predictions_all_list, torch.tensor(clean_verbose_list)), dim=0)
            clean_verbose_listnp = np.array(clean_verbose_list)
            idxsort = np.argsort(-clean_verbose_listnp,axis=1,kind='stable')
            predictions = torch.tensor(idxsort[:,0])
            clean_corr_index = torch.cat((clean_corr_index, predictions.eq(targets)), dim=0)
            wrong_index = torch.cat((wrong_index, ~predictions.eq(targets).cpu()), dim=0)
            result_list += result
            temp_cert_index = torch.tensor([i[0] for i in result]) == 2
            temp_cert_index = temp_cert_index | (torch.tensor([i[0] for i in result]) == 4)
            certified_index = torch.cat((certified_index, temp_cert_index), dim=0)

    clean_corr_index = clean_corr_index[1:].type(torch.bool)
    wrong_index = wrong_index[1:].type(torch.bool)
    certified_index = certified_index[1:].type(torch.bool)
    predictions_all_list = predictions_all_list[B:]
    labels = labels[1:]
    correct_proportion = clean_corr_index.sum().item() / total
    cert_wrong_proportion = (torch.tensor([i[0] for i in result_list]) == 4).sum().item() / total

    cert_proportion = certified_index.sum().item() / total
    cert_correct_proportion = cert_proportion - cert_wrong_proportion

    result = {
        "correct_proportion": correct_proportion,
        "cert_correct_proportion": cert_correct_proportion,
        "cert_wrong_proportion": cert_wrong_proportion,
        "cert_proportion": cert_proportion,
        "clean_corr_index": clean_corr_index,
        "wrong_index": wrong_index,
        "certified_index": certified_index,
        "predictions_all_list": predictions_all_list
    }
    return result

def certify_based_on_bagnet(net, window_size, test_loader, mode="rm"):
    # Certification for BagNet, adapted from https://github.com/inspire-group/PatchGuard/blob/master/mask_bn_cifar.py
    cert_verbose = True
    if mode == "rm": # robust masking
        m = True
        cbn = False
        print("Under mode robust masking")
    else:
        m = False
        cbn = True
        print("under mode CBN")
    accuracy_list=[]
    result_list=[]
    label_list = []
    clean_corr=0
    predictions_all_list = torch.ones(1, num_cls)
    label_list = torch.ones(1)
    for data,labels in (test_loader):
        #original_images = torch.cat((original_images, data), dim=0)
        data = data.to(device)
        label_list = torch.cat((label_list, labels), dim=0)
        labels = labels.numpy()
        with torch.no_grad():
            output_clean = net(data).detach().cpu().numpy() # logits
        #output_clean = softmax(output_clean,axis=-1) # confidence
        #output_clean = (output_clean > 0.2).astype(float) # predictions with confidence threshold
        #note: the provable analysis of robust masking is cpu-intensive and can take some time to finish
        #you can dump the local feature and do the provable analysis with another script so that GPU mempry is not always occupied
        for i in range(len(labels)):
            if m:#robust masking
                result = provable_masking(output_clean[i],labels[i],thres=0,window_shape=[window_size,window_size], verbose=cert_verbose)
                result_list.append(result)
                clean_pred_verbose = masking_defense(output_clean[i],thres=0,window_shape=[window_size,window_size], verbose=True)
                predictions_all_list = torch.cat((predictions_all_list, torch.tensor(clean_pred_verbose).reshape(1, -1)), dim=0)
                clean_verbose_listnp = np.array(clean_pred_verbose)
                idxsort = np.argsort(-clean_verbose_listnp,axis=0,kind='stable')
                clean_pred = idxsort[0]
                clean_corr += clean_pred == labels[i]
            elif cbn:#cbn
                result = provable_clipping(output_clean[i],labels[i],window_shape=[window_size,window_size])
                result_list.append(result)
                clean_pred = clipping_defense(output_clean[i])
                clean_corr += clean_pred == labels[i]
        acc_clean = np.mean(np.argmax(np.mean(output_clean, axis=(1,2)), axis=1) == labels)
        accuracy_list.append(acc_clean)

    # result_list contains the inference result of "certification" part
    # It is in fact different from "robust inference".
    cases,cnt = np.unique([i[0] for i in result_list],return_counts=True)
    #print("Provable robust accuracy:",cnt[1]/len(result_list))
    #print("Clean accuracy with defense:",clean_corr/len(result_list))
    #print("Clean accuracy without defense:",np.mean(accuracy_list))
    #print("------------------------------")
    correct_proportion = clean_corr / len(result_list)
    cert_correct_proportion = cnt[1] / len(result_list)
    cert_wrong_proportion = cnt[3] / len(result_list)
    cert_proportion = cert_correct_proportion + cert_wrong_proportion
    result_index = torch.tensor([i[0] for i in result_list])
    wrong_index = (result_index == 4) | (result_index == 3)
    certified_index = (result_index == 2) | (result_index == 4)
    clean_corr_index = (result_index == 1) | (result_index == 2)
    #if cert_verbose == True:
    #    print("Provable analysis cases (1: Not certified but correct; 2: Certified and correct; 3: Not certified and not correct; 4: Certified but not correct):",cases)
    #else:
    #    print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):",cases)
    print("Provable analysis breakdown",cnt/len(result_list))
    predictions_all_list = predictions_all_list[1:]
    label_list = label_list[1:]

    result = {
        "correct_proportion": correct_proportion,
        "cert_correct_proportion": cert_correct_proportion,
        "cert_wrong_proportion": cert_wrong_proportion,
        "cert_proportion": cert_proportion,
        "clean_corr_index": clean_corr_index,
        "wrong_index": wrong_index,
        "certified_index": certified_index,
        "correct_without_defense": np.mean(accuracy_list),
        "predictions_all_list": predictions_all_list
    }

    return result

"""
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_32_resnet18_band_4_downscale_epoch_199.pth" --rsize 32 --certify_mode ps --patch_size 5 --out_path result/band4_32_ps_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_26_resnet18_band_4_downscale_epoch_199.pth" --rsize 26 --certify_mode ps --patch_size 5 --out_path result/band4_26_ps_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_20_resnet18_band_4_downscale_epoch_199.pth" --rsize 20 --certify_mode ps --patch_size 5 --out_path result/band4_20_ps_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_14_resnet18_band_4_downscale_epoch_199.pth" --rsize 14 --certify_mode ps --patch_size 5 --out_path result/band4_14_ps_5.pt

python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_32_resnet18_band_4_downscale_epoch_199.pth" --rsize 32 --certify_mode pg --patch_size 5 --out_path result/band4_32_pg_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_26_resnet18_band_4_downscale_epoch_199.pth" --rsize 26 --certify_mode pg --patch_size 5 --out_path result/band4_26_pg_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_20_resnet18_band_4_downscale_epoch_199.pth" --rsize 20 --certify_mode pg --patch_size 5 --out_path result/band4_20_pg_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_14_resnet18_band_4_downscale_epoch_199.pth" --rsize 14 --certify_mode pg --patch_size 5 --out_path result/band4_14_pg_5.pt

python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize32_epoch20.pt" --rsize 32 --certify_mode pg --patch_size 30 --out_path result/bagnet17_32_pg_30.pt
python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize26_epoch20.pt" --rsize 26 --certify_mode pg --patch_size 30 --out_path result/bagnet17_26_pg_30.pt
python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize20_epoch20.pt" --rsize 20 --certify_mode pg --patch_size 30 --out_path result/bagnet17_20_pg_30.pt
python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize14_epoch20.pt" --rsize 14 --certify_mode pg --patch_size 30 --out_path result/bagnet17_14_pg_30.pt
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Certification of different methods')
    parser.add_argument('--model_arch', default="resnet", type=str, help='model architecture', choices=["bagnet17", "resnet"])
    parser.add_argument('--model_path', type=str, help="the path for saved model")
    parser.add_argument('--rsize', default=32, type=int, help="the downscale size")
    parser.add_argument('--certify_mode', default="ps", type= str, choices=["ps", "pg"]) # pg: patch guard. ps: patch smoothing.
    parser.add_argument('--patch_size', default=5, type=int, help="the certified patch size")
    parser.add_argument('--batch_size', default=16, type=int, help="the batch size")
    parser.add_argument("--out_path", type=str, help="the output path")
    args = parser.parse_args()

    model_path = args.model_path
    model_arch = args.model_arch
    resize = args.rsize
    batch_size = args.batch_size
    patch_size = args.patch_size

    test_loader = get_test_loader(model_arch, resize, batch_size=batch_size, original_size=32)

    net = load_model(model_arch, model_path)


    if args.certify_mode == "ps":
        assert model_arch != "bagnet"
        # FIXME: we hard code the parameter here.
        band_size = 4
        result = certify_patchSmooth(test_loader, net, batch_size, band_size, patch_size)
    else:
        # FIXME: we hard code the parameter here.
        if model_arch == "resnet":
            band_size = 4
            result = certify_patchGuard(test_loader, net, batch_size, band_size, patch_size)
        else:
            rf_size = 17
            rf_stride = 8
            C, H, W = 3, 192, 192
            window_size = math.ceil((patch_size + rf_size -1) / rf_stride)
            result = certify_based_on_bagnet(net, window_size, test_loader, mode="rm")
    
    torch.save(result, args.out_path)
    print("Certification for {} under certify_mode {}, resize {} finished".format(model_arch, args.certify_mode, resize))
    print("cert_proportion: {}".format(result['cert_proportion']))
    print("cert_wrong_proportion: {}".format(result['cert_wrong_proportion']))
    print("correct_proportion: {}".format(result['correct_proportion']))









#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import advertorch
from datetime import datetime
from vision_transformer import _create_vision_transformer, VisionTransformer
from timm.models import create_model
from data_loaders import *
from utils import setup_device, accuracy, ensure_dir, gen_mask, write_json, print_dict, load_model


def get_train_config():
    parser = argparse.ArgumentParser("Vision Model Train/Fine-tune")

    # basic config
    parser.add_argument("--exp-name", type=str, default="test", help="experiment name")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    # choices=['ViT_b16_224', 'ResNet50_224', 'ResNet20_32', 'ViT_b32_224']
    parser.add_argument("--model-arch", type=str, default="ViT_b16_224", help='model arch to use')
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[32, 224, 384])
    parser.add_argument("--mask-size", type=int, default=-1,
                        help="size of mask patch, should ensure one of the masks can hide the adversarial patch")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--output-dir", type=str, default='output', help='output folder')
    parser.add_argument("--dataset", type=str, default='CIFAR10', help="dataset for evaluation")
    parser.add_argument("--num-classes", type=int, default=10, help="number of classes in dataset")
    parser.add_argument("--num-samples", type=int, default=100, help="number of samples to test")
    parser.add_argument("--attack-method", type=str, default='all', help='the attack method to use')
    config = parser.parse_args()

    # models config
    process_config(config)
    print_dict(vars(config), title='Config')
    return config


def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    # timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name

    # create some important directories to be used for that experiments
    config.result_dir = os.path.join(config.output_dir, 'save', exp_name)
    for dir in [config.result_dir]:
        ensure_dir(dir)

    # save config
    write_json(vars(config), os.path.join(config.result_dir, 'config.json'))
    return config


def generate_adversarial_samples(model, config, device=torch.device('cpu')):
    samples = []
    num_classes = config.num_classes
    attack_method = config.attack_method
    if attack_method == 'all':
        attacks = [
            advertorch.attacks.GradientSignAttack(model),  # One step fast gradient sign method (Goodfellow et al, 2014).
            # advertorch.attacks.FastFeatureAttack(models),  # Fast attack against a target internal representation of a models using gradient descent (Sabour et al.)
            advertorch.attacks.PGDAttack(model),  # The projected gradient descent attack (Madry et al, 2017).
            advertorch.attacks.LinfMomentumIterativeAttack(model),  # The Linf Momentum Iterative Attack Paper: https://arxiv.org/pdf/1710.06081.pdf
            # advertorch.attacks.CarliniWagnerL2Attack(models, num_classes),  # The Carlini and Wagner L2 Attack, https://arxiv.org/abs/1608.04644 (18138 seconds)
            # advertorch.attacks.ElasticNetL1Attack(models, num_classes),  # The ElasticNet L1 Attack, https://arxiv.org/abs/1709.04114 (78576 seconds)
            advertorch.attacks.DDNL2Attack(model),  # The decoupled direction and norm attack (Rony et al, 2018).
            # advertorch.attacks.SinglePixelAttack(models),  # Single Pixel Attack Algorithm 1 in https://arxiv.org/pdf/1612.06299.pdf
            # advertorch.attacks.LocalSearchAttack(models),  # Local Search Attack Algorithm 3 in https://arxiv.org/pdf/1612.06299.pdf
            advertorch.attacks.SpatialTransformAttack(model, num_classes),  # Spatially Transformed Attack (Xiao et al.)
            # advertorch.attacks.JacobianSaliencyMapAttack(models, num_classes),  # Jacobian Saliency Map Attack This includes Algorithm 1 and 3, y must be not None in perturb
        ]
    else:
        attacks = [eval(f'advertorch.attacks.{attack_method}')(model)]
    print(f'generating clean samples ...')
    dl = eval("{}DataLoader".format(config.dataset))(
        data_dir=os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=config.num_samples,
        num_workers=config.num_workers,
        split='validation',
        shuffle=True
    )
    imgs, labels = next(iter(dl))
    for i in range(labels.size(0)):
        samples.append((imgs[i], labels[i], 'clean'))
    imgs = imgs.to(device)
    for attack in attacks:
        attack_method = attack.__class__.__name__
        print(f'generating adversarial samples with {attack_method} ...')
        start_time = time.time()
        adv_imgs = attack.perturb(imgs)
        adv_imgs = adv_imgs.cpu()
        for i in range(labels.size(0)):
            samples.append((adv_imgs[i], labels[i], attack_method))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'adversarial samples generated with {attack_method} in {elapsed_time} seconds')
    return samples


def verify_inference(model, img, img_mask, attn_mask):
    batch_img = img.repeat(attn_mask.size(0), 1, 1, 1)
    batch_img = batch_img.masked_fill(img_mask == 0, 0.)
    # print(tensor.shape, mask.shape)
    if isinstance(model, VisionTransformer):
        out = model(batch_img, attn_mask)
    else:
        out = model(batch_img)
    top1_prob, top1_catid = torch.topk(out, 1)
    preds = top1_catid.flatten()
    count = preds.bincount()
    # print(count)
    majority_pred = count.argmax()
    verified = count[count > 0].size(0) == 1
    return majority_pred, verified, preds


def evaluate(model, samples, config, device=torch.device('cpu')):
    model.eval()
    img_mask, attn_mask = gen_mask(
        img_size=config.image_size, patch_size=16, mask_size=config.mask_size, with_zero_mask=True)
    img_mask, attn_mask = img_mask.to(device), attn_mask.to(device)
    n_total, n_verified, n_correct, n_verified_correct = 0, 0, 0, 0
    start_time = time.time()
    num_samples = len(samples)
    for i, (sample, label, attack) in enumerate(samples):
        try:
            img = sample.unsqueeze(0).to(device)
            majority_pred, verified, preds = verify_inference(model, img, img_mask, attn_mask)
            correct = preds[0].cpu().item() == label
            n_total += 1
            if correct:
                n_correct += 1
            if verified:
                n_verified += 1
            if correct and verified:
                n_verified_correct += 1
            preds = '-' if verified and correct else preds.cpu().data.tolist()
            acc = n_correct * 100 / n_total
            verify_ratio = n_verified * 100 / n_total
            verified_acc = n_verified_correct * 100 / n_total
            print(f'{i + 1:6d}/{num_samples}; '
                  f'correct {n_correct} ({acc:.3f}); '
                  f'verified {n_verified} ({verify_ratio:.3f}); '
                  f'verified_correct {n_verified_correct} ({verified_acc:.3f}); '
                  f'preds {preds}; label {label}; attack {attack}')
        except Exception as e:
            print(f'[WARNING] failed with exception: {e}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    return {
        'n_total': n_total,
        'n_verified': n_verified,
        'n_correct': n_correct,
        'n_verified_correct': n_verified_correct,
        'accuracy': n_correct * 100 / n_total,
        'verified_ratio': n_verified * 100 / n_total,
        'verified_accuracy': n_verified_correct * 100 / n_total,
        'accuracy_in_verified': n_verified_correct * 100 / n_verified,
        'elapsed_time': elapsed_time,
        'elapsed_time_per_sample': elapsed_time / n_total,
        'exp_name': config.exp_name
    }


def main():
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # create model
    print("create model")
    model = load_model(config.model_arch, config.num_classes, config.checkpoint_path)

    # send models to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    samples = generate_adversarial_samples(model, config, device)
    with torch.no_grad():
        result = evaluate(model, samples, config, device)
    print_dict(result, title='Result')
    write_json(result, os.path.join(config.result_dir, 'result.json'))


if __name__ == '__main__':
    main()


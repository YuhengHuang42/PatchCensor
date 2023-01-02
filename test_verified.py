#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import time

import torch

from data_loaders import *
from utils import setup_device, ensure_dir, gen_mask, write_json, print_dict, load_model
from vision_transformer import VisionTransformer

def get_train_config():
    parser = argparse.ArgumentParser("Vision Model Train/Fine-tune")

    # basic config
    parser.add_argument("--exp-name", type=str, default="test", help="experiment name")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    # choices=['ViT_b16_224', 'ResNet50_224', 'ResNet20_32', 'ViT_b32_224']
    parser.add_argument("--model-arch", type=str, default="ViT_b16_224", help='model arch to use')
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[32, 224, 384])
    parser.add_argument("--patch-size", type=int, default=16, help="patch size in ViT", choices=[16, 32])
    parser.add_argument("--mask-size", type=int, default=-1,
                        help="size of mask patch, should ensure one of the masks can hide the adversarial patch")
    parser.add_argument("--use-attn-mask", action='store_true', default=False, help="verify with attention mask")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--output-dir", type=str, default='output', help='output folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for evaluation")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument("--no-verify", action='store_true', default=False, help="test without verification")
    parser.add_argument("--result", type=str, default=None, help="where to save the result")
    parser.add_argument("--skip", default=None, type=int, help="number of example to skip")
    #parser.add_argument("--num_subset", default=None, type=int, help="Only sample a subset of the dataset with length to num_subset")
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


def verify_inference(model, img, img_mask, attn_mask, use_attn_mask=True):
    batch_img = img.repeat(attn_mask.size(0), 1, 1, 1)
    batch_img = batch_img.masked_fill(img_mask == 0, 0.)
    # print(tensor.shape, mask.shape)
    if use_attn_mask and isinstance(model, VisionTransformer):
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


def evaluate(model, config, device=torch.device('cpu')):
    model.eval()
    if config.dataset == "ImageNet":
        dataset_dir = 'imagenet2012'
        split="validation"
        loader_name = config.dataset
        special_data_flag = None
    elif config.dataset == "partImageNet":
        dataset_dir = "PartImageNet"
        split="val"
        loader_name = "ImageNet"
        special_data_flag = "PartImageNet"
    else:
        dataset_dir = config.dataset
        split="test"
        loader_name = config.dataset
        special_data_flag = None
    dl = eval("{}DataLoader".format(loader_name))(
        data_dir=os.path.join(config.data_dir, dataset_dir),
        image_size=config.image_size,
        batch_size=1,
        num_workers=config.num_workers,
        split=split,
        special_data_flag=special_data_flag,
        skip=config.skip
        )
    img_mask, attn_mask = gen_mask(
        img_size=config.image_size,
        patch_size=config.patch_size,
        mask_size=config.mask_size,
        with_zero_mask=True
    )
    img_mask, attn_mask = img_mask.to(device), attn_mask.to(device)
    use_attn_mask = config.use_attn_mask
    n_total, n_verified, n_correct, n_verified_correct = 0, 0, 0, 0
    start_time = time.time()
    file_list = list()
    correct_list = list()
    verified_list = list()
    verified_correct_list = list()
    if isinstance(dl, ImageNetDataLoader) and special_data_flag == None:
        #ds = dl.dataset
        #samples = ds.samples
        num_samples = len(dl)
        # from timm.data import resolve_data_config
        # from timm.data.transforms_factory import create_transform
        # config = resolve_data_config({}, models=models)
        # config['is_training'] = False
        # transform = create_transform(**config)
        #transform = ds.transform
        for i, all_data in enumerate(dl):
            path = all_data[2]
            sample = all_data[0]
            target = all_data[1]
            try:
                file_list.append(path)
                #sample = ds.loader(path)
                #if transform is not None:
                #    sample = transform(sample)
                #img = sample.unsqueeze(0).to(device)
                img = sample.to(device)
                majority_pred, verified, preds = verify_inference(model, img, img_mask, attn_mask, use_attn_mask)
                correct = preds[0].cpu().item() == target
                correct_list.append(correct)
                n_total += 1
                if correct:
                    n_correct += 1
                if verified:
                    n_verified += 1
                if correct and verified:
                    n_verified_correct += 1
                verified_list.append(verified)
                verified_correct_list.append(correct and verified)
                preds = '-' if verified and correct else preds.cpu().data.tolist()
                acc = n_correct * 100 / n_total
                verify_ratio = n_verified * 100 / n_total
                verified_acc = n_verified_correct * 100 / n_total
                print(f'{i + 1:6d}/{num_samples}; '
                      f'correct {n_correct} ({acc:.3f}); '
                      f'verified {n_verified} ({verify_ratio:.3f}); '
                      f'verified_correct {n_verified_correct} ({verified_acc:.3f}); '
                      f'preds {preds}; label {target}; path {path}')
            except Exception as e:
                print(f'[WARNING] failed on {path}: {e}')
    else:
        num_samples = len(dl)
        file_list = []
        for i, all_data in enumerate(dl):
            img = all_data[0]
            label = all_data[1]
            if "imagenet" in config.dataset.lower():
                file_name = all_data[2]
                file_list.append(file_name)
            else:
                file_name = None
            img = img.to(device)
            majority_pred, verified, preds = verify_inference(model, img, img_mask, attn_mask, use_attn_mask)
            target = label[0].item()
            correct = preds[0].cpu().item() == target
            n_total += 1
            correct_list.append(correct)
            if correct:
                n_correct += 1
            verified_list.append(verified)
            if verified:
                n_verified += 1
            verified_correct_list.append(correct and verified)
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
                  f'preds {preds}; label {target}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    save_result = {
        "correct_list": correct_list,
        "verified_list": verified_list,
        "verified_correct_list": verified_correct_list,
        "file_list": file_list
    }
    if config.result is not None:
        torch.save(save_result, config.result)
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


def evaluate_no_verify(model, config, device=torch.device('cpu')):
    model.eval()
    dl = eval("{}DataLoader".format(config.dataset))(
        data_dir=os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=1,
        num_workers=config.num_workers,
        split='validation')
    n_total, n_correct = 0, 0
    start_time = time.time()
    num_samples = len(dl)
    for i, (img, label) in enumerate(dl):
        img = img.to(device)
        out = model(img)
        top1_prob, top1_catid = torch.topk(out, 1)
        pred = top1_catid.flatten()[0]
        target = label[0].item()
        correct = pred == target
        n_total += 1
        if correct:
            n_correct += 1
        acc = n_correct * 100 / n_total
        print(f'{i + 1:6d}/{num_samples}; '
              f'correct {n_correct} ({acc:.3f}); '
              f'pred {pred}; label {target}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    return {
        'n_total': n_total,
        'n_correct': n_correct,
        'accuracy': n_correct * 100 / n_total,
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
    with torch.no_grad():
        if config.no_verify:
            result = evaluate_no_verify(model, config, device)
        else:
            result = evaluate(model, config, device)
    print_dict(result, title='Result')
    write_json(result, os.path.join(config.result_dir, 'result.json'))


if __name__ == '__main__':
    main()


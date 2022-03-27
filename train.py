#!/usr/bin/python
# -*- coding: utf-8 -*-

# code modified from https://github.com/asyml/vision-transformer-pytorch/

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter, ensure_dir, gen_mask, print_dict, load_model
from vision_transformer import VisionTransformer


def get_train_config():
    parser = argparse.ArgumentParser("Vision Model Train/Fine-tune")

    # basic config
    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--tensorboard", default=False, action='store_true', help='flag of turnning on tensorboard')
    # choices=['ViT_b16_224', 'ResNet50_224', 'ResNet20_32', 'ViT_b32_224']
    parser.add_argument("--model-arch", type=str, default="ViT_b16_224", help='model arch to use')
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[32, 224, 384])
    parser.add_argument("--mask-size", type=int, default=-1, help="mask size to use in training")
    parser.add_argument("--mask-width", type=int, default=-1, help="mask width to use in VisionTransformer training")
    parser.add_argument("--use-attn-mask", action='store_true', default=False, help="train and verify with attention mask")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--train-epochs", type=int, default=5, help="number of training/fine-tunning epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--output-dir", type=str, default='output', help='output folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    config = parser.parse_args()

    # models config
    process_config(config)
    print_dict(vars(config), 'Config')
    return config


def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    # timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    # exp_name = config.exp_name + '_{}_bs{}_lr{}_wd{}'.format(config.dataset, config.batch_size, config.lr, config.wd)
    # exp_name += ('_' + timestamp)
    exp_name = config.exp_name

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join(config.output_dir, 'tb', exp_name)
    config.checkpoint_dir = os.path.join(config.output_dir, 'save', exp_name, 'checkpoints/')
    config.result_dir = os.path.join(config.output_dir, 'save', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.result_dir]:
        ensure_dir(dir)

    # save config
    # write_json(vars(config), os.path.join('experiments', 'save', exp_name, 'config.json'))
    return config


def train_epoch(epoch, model, data_loader, criterion, optimizer,
                lr_scheduler, metrics, device=torch.device('cpu'), masks=None):
    metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        if masks is not None and isinstance(model, VisionTransformer):
            batch_size = batch_data.size(0)
            img_mask, attn_mask = masks
            num_masks = img_mask.size(0)
            random_idx = torch.randint(0, num_masks, size=(batch_size,))
            batch_img_mask = img_mask[random_idx]
            batch_attn_mask = attn_mask[random_idx]
            batch_data = batch_data.masked_fill(batch_img_mask == 0, 0.)
            batch_pred = model(batch_data, batch_attn_mask)
        else:
            batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        metrics.update('acc1', acc1.item())
        metrics.update('acc5', acc5.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, Acc@5: {:.2f}"
                    .format(epoch, batch_idx, len(data_loader), loss.item(), acc1.item(), acc5.item()))
    return metrics.result()


def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu'), masks=None):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            if masks is not None and isinstance(model, VisionTransformer):
                batch_size = batch_data.size(0)
                img_mask, attn_mask = masks
                num_masks = img_mask.size(0)
                random_idx = torch.randint(0, num_masks, size=(batch_size,))
                batch_img_mask = img_mask[random_idx]
                batch_attn_mask = attn_mask[random_idx]
                batch_data = batch_data.masked_fill(batch_img_mask == 0, 0.)
                batch_pred = model(batch_data, batch_attn_mask)
            else:
                batch_pred = model(batch_data)

            loss = criterion(batch_pred, batch_target)
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('acc5', acc5)
    return metrics.result()


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False, save_epochs=True):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'current.pth')
    if save_epochs:
        filename = str(save_dir + f'epoch_{epoch}.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'best.pth')
        torch.save(state, filename)


def main():
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    model = load_model(config.model_arch, config.num_classes, config.checkpoint_path)

    # send models to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    print("create dataloaders")
    dataset_dir = config.dataset
    if dataset_dir == 'ImageNet':
        dataset_dir = 'imagenet2012'
    train_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, dataset_dir),
                    image_size=config.image_size,
                    mask_size=config.mask_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='train')
    valid_dataloader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, dataset_dir),
                    image_size=config.image_size,
                    mask_size=config.mask_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='validation')

    # training criterion
    print("create criterion and optimizer")
    criterion = nn.CrossEntropyLoss()

    # create optimizers and learning rate scheduler
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        epochs=config.train_epochs,
        steps_per_epoch = len(train_dataloader)
    )

    use_attn_mask = config.use_attn_mask
    if use_attn_mask and config.mask_width > 0 and isinstance(model, VisionTransformer):
        img_mask, attn_mask = gen_mask(img_size=224, patch_size=16, mask_width=config.mask_width)
        img_mask, attn_mask = img_mask.to(device), attn_mask.to(device)
        masks = [img_mask, attn_mask]
    else:
        masks = None

    # start training
    print("start training")
    best_acc = 0.0
    epochs = config.train_epochs
    for epoch in range(1, epochs + 1):
        log = {'epoch': epoch}

        # train the models
        model.train()
        result = train_epoch(epoch, model, train_dataloader, criterion,
                             optimizer, lr_scheduler, train_metrics, device, masks=masks)
        log.update(result)

        # validate the models
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, criterion,
                             valid_metrics, device, masks=masks)
        log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        best = False
        if log['val_acc1'] > best_acc:
            best_acc = log['val_acc1']
            best = True

        # save models
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    main()


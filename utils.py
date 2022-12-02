#!/usr/bin/python
# -*- coding: utf-8 -*-

# code modified from https://github.com/asyml/vision-transformer-pytorch/blob/main/src/utils.py

import os
import json
import pandas as pd
import torch
from vision_transformer import _create_vision_transformer
from timm.models import create_model
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import importlib


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=2, sort_keys=False)


def print_dict(dict_data, title='Info'):
    message = ''
    message += f'----------------- {title} ---------------\n'
    for k, v in sorted(dict_data.items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res


def gen_mask(img_size=224, patch_size=16, mask_size=32, mask_width=None, with_zero_mask=False):
    n = int(img_size / patch_size)
    n_total_patches = n * n + 1
    mw = mask_width if (mask_width is not None and mask_width > 0) else int(mask_size / patch_size)
    img_masks = []
    attn_masks = []
    if with_zero_mask:
        img_masks.append(torch.ones(size=(img_size, img_size)))
        attn_masks.append(torch.ones(size=(n_total_patches, n_total_patches)))
    print(f'generating mask: n={n} mask_width={mw} patch_size={patch_size}')
    for i in range(n - mw + 1):
        for j in range(n - mw + 1):
            img_mask = torch.ones(size=(img_size, img_size))
            img_mask[i*patch_size:(i+mw)*patch_size, j*patch_size:(j+mw)*patch_size] = 0
            img_masks.append(img_mask)

            attn_mask = torch.ones(size=(n_total_patches, n_total_patches))
            mat = torch.zeros(size=(n, n))
            mat[i:i+mw, j:j+mw] = 1
            masked_idx = mat.flatten().nonzero() + 1
            attn_mask[:, masked_idx] = 0
            attn_mask[masked_idx, :] = 0
            # print(f'masked patch: [{i}:{i+ps-1}]x[{j}:{j+ps-1}]')
            attn_masks.append(attn_mask)
    batch_img_mask = torch.stack(img_masks).unsqueeze(1)
    batch_attn_mask = torch.stack(attn_masks).unsqueeze(1)
    print(f'batch_img_mask:{batch_img_mask.shape}, batch_attn_mask:{batch_attn_mask.shape}')
    return batch_img_mask, batch_attn_mask


def gen_mask_batch(masks, batch_size):
    num_masks = masks.size(0)
    random_idx = torch.randint(0, num_masks, size=(batch_size,))
    return masks[random_idx]


def load_checkpoint(path):
    assert path.endswith('pth')
    state_dict = torch.load(path)['state_dict']
    return state_dict


def load_model(model_arch, num_classes, checkpoint_path):
    if model_arch == 'ViT_b16_224':
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        model = _create_vision_transformer('vit_base_patch16_224',
                                           pretrained=True, num_classes=num_classes, **model_kwargs)
    elif model_arch == 'ViT_l16_224':
        model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
        model = _create_vision_transformer('vit_large_patch16_224',
                                           pretrained=True, num_classes=num_classes, **model_kwargs)
    elif model_arch == 'ViT_b32_384':
        model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
        model = _create_vision_transformer('vit_base_patch32_384',
                                           pretrained=True, num_classes=num_classes, **model_kwargs)
    elif model_arch == 'ResNet50_224':
        model = create_model(model_name='swsl_resnet50', pretrained=True, num_classes=num_classes)
    # elif model_arch == 'ResNet20_32':
    #     model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    elif model_arch == 'ViT_b32_224':
        model = create_model(model_name="vit_base_patch32_224", pretrained=True, num_classes=num_classes)
    else:
        print(f'unknown model arch: {model_arch}')
        return None

    # load checkpoint
    if checkpoint_path:
        state_dict = load_checkpoint(checkpoint_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(checkpoint_path))

    return model


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


def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}_bs{}_lr{}_wd{}'.format(config.dataset, config.batch_size, config.lr, config.wd)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join('experiments', 'tb', exp_name)
    config.checkpoint_dir = os.path.join('experiments', 'save', exp_name, 'checkpoints/')
    config.result_dir = os.path.join('experiments', 'save', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.result_dir]:
        ensure_dir(dir)

    # save config
    write_json(vars(config), os.path.join('experiments', 'save', exp_name, 'config.json'))

    return config


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                print(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    if name == 'add_embedding':
                        add_data(tag=tag, mat=data, global_step=self.step, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


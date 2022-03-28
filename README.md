## About 

This is the source code of PatchCensor.
Including the scripts to reproduce the results in the paper.

## Prerequisites

The code was tested on Python 3.6 and PyTorch 1.4.0.
The pre-trained models are obtained from the timm project (https://github.com/rwightman/pytorch-image-models).

The timm package must be installed by:

```bash
git clone https://github.com/rwightman/pytorch-image-models.git
pip install -e pytorch-image-models
```


## How to use

`train.py` is the script for training/fine-tuning models.
The Minority Report defense requires to train models for different patch sizes.

Example usage:

```bash
python train.py --exp-name train_ResNet50_CIFAR10_mask32
        --n-gpu 4 --model-arch ResNet50 --dataset CIFAR10
        --num-classes 1000 --image-size 224 --mask-size 32
        --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchcensor
```

`test_verified` is the script to test the verified detection performance of different models.

Example usage:

```bash
python -u test_verified.py --exp-name vit_imagenet_val_mask3_verify \
        --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 \
        --patch-size 16  --mask-size 48 --checkpoint-path /mnt/mydrive/output/patchcensor/save/train_vit_mask_width3/checkpoints/best.pth \
        --data-dir /mnt/mydrive/data --output-dir ./output/patchcensor
```


Detailed commands to setup environment and run experiments can be found in `test_script.sh`.

## Acknowledgement

We refer to [PatchGuard](https://github.com/inspire-group/PatchGuard) for the implementation of PatchGuard and [PatchSmoothing](https://github.com/alevine0/patchSmoothing)
for the implementation of (De)Randomized Smoothing for Certifiable Defense against Patch Attacks.


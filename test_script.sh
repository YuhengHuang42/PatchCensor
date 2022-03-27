# environment setup
pip install dataclasses
git clone https://github.com/rwightman/pytorch-image-models.git
pip install -e pytorch-image-models

# CIFAR10 experiments

# fine-tune ImageNet models on CIFAR10
python train.py --exp-name vit_cifar --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 0 < adv_size < 16, 0% < adv_pixels < 0.51%
python train.py --exp-name vit_cifar_mask2 --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --mask-width 2 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar_mask2 --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --patch-size 32 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask2_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask2_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask2/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask2_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask2_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar_mask2/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 16 < adv_size < 32, 0.51% < adv_pixels < 2.04%
python train.py --exp-name vit_cifar_mask3 --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --mask-width 3 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar_mask3 --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --patch-size 48 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask3_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask3_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask3/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask3_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask3_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar_mask3/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 32 < adv_size < 48, 2.04% < adv_pixels < 4.59%
python train.py --exp-name vit_cifar_mask4 --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --mask-width 4 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar_mask4 --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --patch-size 64 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask4_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask4_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask4/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask4_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask4_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar_mask4/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 48 < adv_size < 64, 4.59% < adv_pixels < 8.16%
python train.py --exp-name vit_cifar_mask5 --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --mask-width 5 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar_mask5 --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --patch-size 80 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask5_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask5_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask5/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask5_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask5_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar_mask5/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 64 < adv_size < 80, 8.16% < adv_pixels < 12.76%
python train.py --exp-name vit_cifar_mask6 --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --mask-width 6 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar_mask6 --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --patch-size 96 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask6_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask6_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask6/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask6_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask6_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar_mask6/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 80 < adv_size < 96, 12.76% < adv_pixels < 18.37%
python train.py --exp-name vit_cifar_mask7 --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --mask-width 7 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar_mask7 --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --patch-size 112 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask7_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask7_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask7/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask7_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask7_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar_mask7/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 96 < adv_size < 112, 18.37% < adv_pixels < 25.00%
python train.py --exp-name vit_cifar_mask8 --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --mask-width 8 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_cifar_mask8 --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --patch-size 128 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask8_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask8_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask8/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask8_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_cifar_mask8_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_cifar_mask8/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto


# CIFAR10 verify only
python test_verified.py --exp-name vit_cifar_mask2_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask3_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask4_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask5_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask6_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask7_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask8_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# CIFAR10 ft verify
python test_verified.py --exp-name vit_cifar_mask2_ft_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask2/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask3_ft_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask3/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask4_ft_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask4/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask5_ft_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask5/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask6_ft_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask6/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask7_ft_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask7/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_cifar_mask8_ft_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset CIFAR10 --num-classes 10 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/vit_cifar_mask8/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto


# ImageNet experiments

# 0 < adv_size < 16, 0% < adv_pixels < 0.51%
python test_verified.py --exp-name vit_imagenet_mask2_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 15 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_imagenet_mask2 --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --patch-size 32 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto --train-steps 100000
python test_verified.py --exp-name resnet_imagenet_mask2_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask2/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 16 < adv_size < 32, 0.51% < adv_pixels < 2.04%
python test_verified.py --exp-name vit_imagenet_mask3_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_imagenet_mask3 --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --patch-size 48 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto --train-steps 100000
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 32 < adv_size < 48, 2.04% < adv_pixels < 4.59%
python test_verified.py --exp-name vit_imagenet_mask4_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 47 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_imagenet_mask4 --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --patch-size 64 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto --train-steps 100000
python test_verified.py --exp-name resnet_imagenet_mask4_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask4/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 48 < adv_size < 64, 4.59% < adv_pixels < 8.16%
python test_verified.py --exp-name vit_imagenet_mask5_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 63 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_imagenet_mask5 --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --patch-size 80 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto --train-steps 100000
python test_verified.py --exp-name resnet_imagenet_mask5_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask5/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 64 < adv_size < 80, 8.16% < adv_pixels < 12.76%
python test_verified.py --exp-name vit_imagenet_mask6_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 79 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_imagenet_mask6 --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --patch-size 96 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto --train-steps 100000
python test_verified.py --exp-name resnet_imagenet_mask6_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask6/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 80 < adv_size < 96, 12.76% < adv_pixels < 18.37%
python test_verified.py --exp-name vit_imagenet_mask7_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 95 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_imagenet_mask7 --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --patch-size 112 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto --train-steps 100000
python test_verified.py --exp-name resnet_imagenet_mask7_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask7/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# 96 < adv_size < 112, 18.37% < adv_pixels < 25.00%
python test_verified.py --exp-name vit_imagenet_mask8_verify --n-gpu 4 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 111 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python train.py --exp-name resnet_imagenet_mask8 --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --patch-size 128 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto --train-steps 100000
python test_verified.py --exp-name resnet_imagenet_mask8_ft_verify --n-gpu 4 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask8/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto


# imagenet verify using vit
python test_verified.py --exp-name vit_imagenet_mask2_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 15 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_imagenet_mask3_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_imagenet_mask4_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 47 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_imagenet_mask5_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 63 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_imagenet_mask6_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 79 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_imagenet_mask7_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 95 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name vit_imagenet_mask8_verify --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 111 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# imagenet verify using resnet with fine-tuning
python test_verified.py --exp-name resnet_imagenet_mask2_ft_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 15 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask2_20w/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_20w/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask4_ft_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 47 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask4_20w/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask5_ft_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 63 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask5_20w/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask6_ft_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 79 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask6_20w/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask7_ft_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 95 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask7_20w/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask8_ft_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 111 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask8_20w/checkpoints/best.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# imagenet verify using resnet without fine-tuning
python test_verified.py --exp-name resnet_imagenet_mask2_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 15 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask4_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 47 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask5_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 63 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask6_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 79 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask7_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 95 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask8_verify --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 111 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto

# imagenet verify using resnet with different fine-tuning epochs
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch3 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_3.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch6 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_6.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch9 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_9.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch12 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_12.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch15 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_15.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch18 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_18.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch21 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_21.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto
python test_verified.py --exp-name resnet_imagenet_mask3_ft_verify_epoch24 --n-gpu 1 --model-arch ResNet50_224 --dataset ImageNet --num-classes 1000 --image-size 224 --adv-size 31 --checkpoint-path /mnt/mydrive/output/patchveto/save/resnet_imagenet_mask3_100w/checkpoints/epoch_24.pth --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto


# empirical evaluation on adversarial perturbations
python test_perturbation.py --exp-name empirical_adv_perturbation --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --data-dir /mnt/mydrive/data --output-dir /mnt/mydrive/output/patchveto


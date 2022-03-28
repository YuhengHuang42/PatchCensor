# environment setup
pip install dataclasses
git clone https://github.com/rwightman/pytorch-image-models.git
pip install -e pytorch-image-models

# Fine-tuning VIT
python -u train.py --exp-name train_vit_mask_width3 \
        --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet \
        --num-classes 1000 --image-size 224 --mask-width 3 \
        --data-dir [YOUR DATA DIR] --output-dir YOUR OUTPUT] \

# Evaluation result on PartImagenet
python -u test_verified.py --exp-name vit_imagenet_mask3_verify \
--model-arch ViT_b16_224 --dataset partImageNet --num-classes 1000 \
--patch-size 16  --mask-size 48 --checkpoint-path [MODEL PATH FROM FINE-TUNING] \
--data-dir [YOUR DATA DIR] --output-dir [YOUR OUTPUT]

# Evaluate on ImageNet with SKIP=10
python -u test_verified.py --exp-name vit_imagenet_val_mask3_verify \
--model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 \
--patch-size 16  --mask-size 48 --checkpoint-path [MODEL PATH FROM FINE-TUNING] \
--data-dir [YOUR DATA DIR] --output-dir YOUR OUTPUT]--skip 10 --result result_vit_imagenet_val.pt

# test_verified.py also work with other datasets like CIFAR-10 and other architectures like ResNet
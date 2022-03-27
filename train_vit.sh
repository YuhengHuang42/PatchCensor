#python -u train.py --exp-name train_vit_mask_width3 \
#        --n-gpu 1 --model-arch ViT_b16_224 --dataset ImageNet \
#        --num-classes 1000 --image-size 224 --mask-width 3 \
#        --data-dir /MALEI/ --output-dir ./output/patchveto \

#python -u test_verified.py --exp-name vit_imagenet_mask3_verify \
#--model-arch ViT_b16_224 --dataset partImageNet --num-classes 1000 \
#--patch-size 16  --mask-size 48 --checkpoint-path /home/malei/yuheng_workspace/voterStudy/output/patchveto/save/train_vit_mask32/checkpoints/best.pth \
#--data-dir "/MALEI/partImageNet/" --output-dir ./output/patchveto

python -u test_verified.py --exp-name vit_imagenet_val_mask3_verify \
--model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 \
--patch-size 16  --mask-size 48 --checkpoint-path /home/malei/yuheng_workspace/voterStudy/output/patchveto/save/train_vit_mask_width3/checkpoints/best.pth \
--data-dir "/MALEI/" --output-dir ./output/patchveto 
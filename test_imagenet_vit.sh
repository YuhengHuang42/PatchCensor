data_dir="/data/data_disk/"
output_dir="results/vit_finetune_transformer"

#python3 test_verified.py --exp-name vit_imagenet_mask8_ft_verify --n-gpu 2 --model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 --mask-size 128 --data-dir $data_dir --output-dir $output_dir \
#--checkpoint-path /home/huangyuheng/PatchCensor/results/vit_finetune_transformer/save/vit_imagenet_mask128/checkpoints/best.pth

#python3 test_verified.py --exp-name vit_imagenet_mask7_ft_verify --n-gpu 2 \
#--model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 \
#--mask-size 112 --data-dir $data_dir --output-dir $output_dir \
#--checkpoint-path /home/huangyuheng/PatchCensor/results/vit_finetune_transformer/save/vit_imagenet_mask112/checkpoints/best.pth

python3 test_verified.py --exp-name vit_imagenet_mask6_ft_verify --n-gpu 2 \
--model-arch ViT_b16_224 --dataset ImageNet --num-classes 1000 --image-size 224 \
--mask-size 96 --data-dir $data_dir --output-dir $output_dir \
--checkpoint-path /home/huangyuheng/PatchCensor/results/vit_finetune_transformer/save/vit_imagenet_mask96/checkpoints/best.pth
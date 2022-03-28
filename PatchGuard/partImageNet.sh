
#python -u mask_bn.py --data_dir "/MALEI/partImageNet/" --model bagnet17 --dataset PartImageNet --patch_size 32 --m #mask-bn with bagnet17 on cifar
#python -u mask_ds.py --data_dir "/MALEI/partImageNet/" --dataset PartImageNet --patch_size 32 --ds #ds for imagenet

python -u mask_bn.py --data_dir "drive/imagenet2012/validation/" --model bagnet17 --dataset imagenet --patch_size 32 --m --skip 10 #mask-bn with bagnet17 on cifar
python -u mask_ds.py --data_dir "drive/imagenet2012/validation/" --dataset imagenet --patch_size 32 --ds --skip 10 #ds for imagenet
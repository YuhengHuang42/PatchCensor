python3 -u train_cifar.py --resize 28 --checkpoint bagNet17_resize28_epoch20.pt > bagnet_resize_28_train.log 2>&1
python3 -u train_cifar.py --resize 24 --checkpoint bagNet17_resize24_epoch20.pt > bagnet_resize_24_train.log 2>&1
python3 -u train_cifar.py --resize 22 --checkpoint bagNet17_resize22_epoch20.pt > bagnet_resize_22_train.log 2>&1

python3 -u train_cifar.py --resize 28 --lr 0.001 --checkpoint adv_bagNet17_resize28_epoch20.pt --resume_path bagNet17_resize28_epoch20.pt --resume --mode adv > adv_bagnet_resize_28_train.log 2>&1
python3 -u train_cifar.py --resize 24 --lr 0.001 --checkpoint adv_bagNet17_resize24_epoch20.pt --resume_path bagNet17_resize24_epoch20.pt --resume --mode adv > adv_bagnet_resize_24_train.log 2>&1
python3 -u train_cifar.py --resize 22 --lr 0.001 --checkpoint adv_bagNet17_resize22_epoch20.pt --resume_path bagNet17_resize22_epoch20.pt --resume --mode adv > adv_bagnet_resize_22_train.log 2>&1

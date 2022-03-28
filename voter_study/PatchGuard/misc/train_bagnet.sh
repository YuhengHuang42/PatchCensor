python3 -u train_cifar.py --resize 32 --checkpoint bagNet17_resize32_epoch20.pt > bagnet_resize_32_train.log 2>&1
python3 -u train_cifar.py --resize 26 --checkpoint bagNet17_resize26_epoch20.pt > bagnet_resize_26_train.log 2>&1
python3 -u train_cifar.py --resize 20 --checkpoint bagNet17_resize20_epoch20.pt > bagnet_resize_20_train.log 2>&1
python3 -u train_cifar.py --resize 14 --checkpoint bagNet17_resize14_epoch20.pt > bagnet_resize_14_train.log 2>&1

python3 -u train_cifar.py --resize 32 --lr 0.001 --checkpoint adv_bagNet17_resize32_epoch20.pt --resume_path bagNet17_resize32_epoch20.pt --resume --mode adv > adv_bagnet_resize_32_train.log 2>&1
python3 -u train_cifar.py --resize 26 --lr 0.001 --checkpoint adv_bagNet17_resize26_epoch20.pt --resume_path bagNet17_resize26_epoch20.pt --resume --mode adv > adv_bagnet_resize_26_train.log 2>&1
python3 -u train_cifar.py --resize 20 --lr 0.001 --checkpoint adv_bagNet17_resize20_epoch20.pt --resume_path bagNet17_resize20_epoch20.pt --resume --mode adv > adv_bagnet_resize_20_train.log 2>&1
python3 -u train_cifar.py --resize 14 --lr 0.001 --checkpoint adv_bagNet17_resize14_epoch20.pt --resume_path bagNet17_resize14_epoch20.pt --resume --mode adv > adv_bagnet_resize_14_train.log 2>&1

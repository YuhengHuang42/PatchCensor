python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_32_resnet18_band_4_downscale_epoch_199.pth" --rsize 32 --certify_mode pg --patch_size 5 --out_path result/band4_32_pg_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_26_resnet18_band_4_downscale_epoch_199.pth" --rsize 26 --certify_mode pg --patch_size 5 --out_path result/band4_26_pg_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_20_resnet18_band_4_downscale_epoch_199.pth" --rsize 20 --certify_mode pg --patch_size 5 --out_path result/band4_20_pg_5.pt
#python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_14_resnet18_band_4_downscale_epoch_199.pth" --rsize 14 --certify_mode pg --patch_size 5 --out_path result/band4_14_pg_5.pt

python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_28_resnet18_band_4_downscale_epoch_199.pth" --rsize 28 --certify_mode pg --patch_size 5 --out_path result/band4_28_pg_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_24_resnet18_band_4_downscale_epoch_199.pth" --rsize 24 --certify_mode pg --patch_size 5 --out_path result/band4_24_pg_5.pt
python3 AOI_study_certify.py --model_arch resnet --model_path "checkpoints/cifar_lr_0.1_model_resize_22_resnet18_band_4_downscale_epoch_199.pth" --rsize 22 --certify_mode pg --patch_size 5 --out_path result/band4_22_pg_5.pt

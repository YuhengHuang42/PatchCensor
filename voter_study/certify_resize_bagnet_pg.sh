#python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize32_epoch20.pt" --rsize 32 --certify_mode pg --patch_size 30 --out_path result/bagnet17_32_pg_30.pt
#python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize26_epoch20.pt" --rsize 26 --certify_mode pg --patch_size 30 --out_path result/bagnet17_26_pg_30.pt
#python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize20_epoch20.pt" --rsize 20 --certify_mode pg --patch_size 30 --out_path result/bagnet17_20_pg_30.pt
#python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize14_epoch20.pt" --rsize 14 --certify_mode pg --patch_size 30 --out_path result/bagnet17_14_pg_30.pt

python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize28_epoch20.pt" --rsize 28 --certify_mode pg --patch_size 30 --out_path result/bagnet17_28_pg_30.pt
python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize24_epoch20.pt" --rsize 24 --certify_mode pg --patch_size 30 --out_path result/bagnet17_24_pg_30.pt
python3 AOI_study_certify.py --model_arch bagnet17 --model_path "checkpoints/adv_bagNet17_resize22_epoch20.pt" --rsize 22 --certify_mode pg --patch_size 30 --out_path result/bagnet17_22_pg_30.pt

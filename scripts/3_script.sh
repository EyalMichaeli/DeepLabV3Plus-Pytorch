
# train base model from scratch


nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_subset_2k_seed_1_munit_aug_ratio_0.25_2023_0603_1025_25_style_std_1.85_lpips_filter_0.1_0.5 \
    --train_sample_ratio 0.67 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/2023_0603_1025_25_ampO1_lower_LR_lower_res_inference_inference_cp_000450000_style_std_1.85_lpips_filter_0.1_0.5.json \
    --aug_sample_ratio 0.25 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2 --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &

wait

nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_subset_2k_seed_1_munit_aug_ratio_0.25_2023_0518_1805_39 \
    --train_sample_ratio 0.67 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cs2cs-style_recon_2_perceptual_1/2023_0518_1805_39_ampO1_lower_LR/inference_cp_400k_style_std_1.5.json \
    --aug_sample_ratio 0.25 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2 --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &

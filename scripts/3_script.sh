
# train base model from scratch

nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/voc/base_seed_1_use_data_0.5 \
    --train_sample_ratio 0.5 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000' \
        2>&1 | tee -a nohup_outputs/voc/output.log &

wait

nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_subset_2k_seed_1_aug_ratio_0.15_ip2p_hive_sd_1.5_rw_2x_image_w_1.5_blip_gpt_v1_ratio_1.0_images_lpips_filter_0.1_0.7 \
    --train_sample_ratio 0.67 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/cityscapes_2023_0710_1646_18_cityscapes_ip2p_hive_sd_1.5_rw_2x_image_w_1.5_blip_gpt_v1_ratio_1.0_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.15 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2 --crop_size 256 --batch_size 32 \
        --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &

wait

nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_subset_2k_seed_1_aug_ratio_0.1_ip2p_hive_sd_1.5_rw_2x_image_w_1.5_blip_gpt_v1_ratio_1.0_images_lpips_filter_0.1_0.7 \
    --train_sample_ratio 0.67 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/cityscapes_2023_0710_1646_18_cityscapes_ip2p_hive_sd_1.5_rw_2x_image_w_1.5_blip_gpt_v1_ratio_1.0_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.1 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2 --crop_size 256 --batch_size 32 \
        --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &

# nohup sh -c 'python main.py \
#     --gpu_id 3 \
#     --random_seed 1 \
#     --logdir logs/cityscapes/cs_subset_2k_seed_1_base \
#     --train_sample_ratio 0.67 \
#         --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
#             --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
#             --total_itrs 30000 \
#             2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &
            
# wait

# nohup sh -c 'python main.py \
#     --gpu_id 3 \
#     --random_seed 2 \
#     --logdir logs/cityscapes/cs_subset_2k_seed_2_base \
#     --train_sample_ratio 0.67 \
#         --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
#             --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
#             --total_itrs 30000 \
#             2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &

# nohup sh -c 'python main.py \
#     --gpu_id 3 \
#     --random_seed 3 \
#     --logdir logs/cityscapes/cs_subset_2k_seed_3_base \
#     --train_sample_ratio 0.67 \
#         --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
#             --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
#             --total_itrs 30000 \
#             2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &


# wait

# nohup sh -c 'python main.py \
#     --gpu_id 0 \
#     --random_seed 1 \
#     --logdir logs/cityscapes/cs_subset_2k_seed_1_aug_ratio_0.15_ip2p__2x_constant_instructions_image_w_1.5_images_masked_person_rider_min_blob_size_1562.5_v0_lpips_filter_0.1_0.6.json \
#     --train_sample_ratio 0.67 \
#     --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/2023_0708_2338_27_cityscapes_ip2p__2x_constant_instructions_image_w_1.5_images_masked_person_rider_min_blob_size_1562.5_v0_lpips_filter_0.1_0.6.json \
#     --aug_sample_ratio 0.15 \
#         --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2 --crop_size 256 --batch_size 32 \
#             --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
#             2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &

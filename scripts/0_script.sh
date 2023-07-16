nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/voc/seed_1_aug_ratio_0.15_pascal_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_2x_image_w_1.5_blip_gpt_v1_ratio_0.3_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.75 \
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_2023_0714_2233_36_pascal_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_2x_image_w_1.5_blip_gpt_v1_ratio_0.3_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.15 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000 --limit_aug_per_image 1' \
        2>&1 | tee -a nohup_outputs/voc/output.log &

wait

nohup sh -c 'python main.py \
    --gpu_id 0 \
    --random_seed 1 \
    --logdir logs/voc/seed_1_aug_ratio_0.25_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.75 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/pascal_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6.json \
    --aug_sample_ratio 0.25 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000' \
        2>&1 | tee -a nohup_outputs/voc/output.log &

# wait

# nohup sh -c 'python main.py \
#     --gpu_id 0 \
#     --random_seed 1 \
#     --logdir logs/cityscapes/cs_subset_2k_seed_1_base \
#     --train_sample_ratio 0.67 \
#         --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
#             --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
#             --total_itrs 30000 \
#             2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &
            
# wait

# nohup sh -c 'python main.py \
#     --gpu_id 0 \
#     --random_seed 2 \
#     --logdir logs/cityscapes/cs_subset_2k_seed_2_base \
#     --train_sample_ratio 0.67 \
#         --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
#             --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
#             --total_itrs 30000 \
#             2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &
nohup sh -c 'python main.py \
    --gpu_id 1 \
    --random_seed 1 \
    --logdir logs/voc/seed_1_use_data_0.5_aug_ratio_0.15_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.5 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/pascal_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6.json \
    --aug_sample_ratio 0.15 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000' \
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
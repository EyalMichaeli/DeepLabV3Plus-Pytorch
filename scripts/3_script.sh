export CUDA_VISIBLE_DEVICES=3

# train from scratch with aug, ip2p
nohup sh -c 'python main.py \
    --gpu_id 0 \
    --random_seed 3 \
    --logdir logs/voc/seed_3_aug_ratio_0.35_pascal_ip2p_2x_constant_instructions_image_w_1.5_with_blip_and_gpt_images_masked__min_blob_size_100000_v0_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.75 \
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_2x_constant_instructions_image_w_1.5_with_blip_and_gpt_images_masked__min_blob_size_100000_v0_lpips_filter_0.1_0.6.json \
    --sample_aug_ratio 0.35 \
    --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results' \
        2>&1 | tee -a nohup_outputs/voc/output.log &

wait
# train from scratch with aug, ip2p
nohup sh -c 'python main.py \
    --gpu_id 0 \
    --random_seed 3 \
    --logdir logs/voc/seed_3_aug_ratio_0.5_pascal_ip2p_2x_constant_instructions_image_w_1.5_with_blip_and_gpt_images_masked__min_blob_size_100000_v0_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.75 \
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_2x_constant_instructions_image_w_1.5_with_blip_and_gpt_images_masked__min_blob_size_100000_v0_lpips_filter_0.1_0.6.json \
    --sample_aug_ratio 0.5 \
    --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results' \
        2>&1 | tee -a nohup_outputs/voc/output.log &


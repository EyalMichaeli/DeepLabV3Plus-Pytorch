nohup sh -c 'python main.py \
    --gpu_id 1 \
    --random_seed 1 \
    --logdir logs/voc/seed_1_aug_ratio_0.15_pascal_ip2p_2x_constant_instructions_image_w_1.5_images_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.75 \
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_2x_constant_instructions_image_w_1.5_images_lpips_filter_0.1_0.6.json \
    --aug_sample_ratio 0.15 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000' \
        2>&1 | tee -a nohup_outputs/voc/output.log &

wait

nohup sh -c 'python main.py \
    --gpu_id 1 \
    --random_seed 1 \
    --logdir logs/voc/seed_1_aug_ratio_0.25_pascal_ip2p_2x_constant_instructions_image_w_1.5_images_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.75 \
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_2x_constant_instructions_image_w_1.5_images_lpips_filter_0.1_0.6.json \
    --aug_sample_ratio 0.25 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000' \
        2>&1 | tee -a nohup_outputs/voc/output.log &
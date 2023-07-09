
nohup sh -c 'python main.py \
    --gpu_id 1 \
    --random_seed 3 \
    --logdir logs/cityscapes/cs_subset_2k_seed_3_base \
    --train_sample_ratio 0.67 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            --total_itrs 30000 \
            2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &
            
wait

# nohup sh -c 'python main.py \
#     --gpu_id 0 \
#     --random_seed 1 \
#     --logdir logs/cityscapes/cs_subset_2k_seed_1_base_lower_lr_and_bs_as_in_repo \
#     --train_sample_ratio 0.67 \
#         --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.1  --crop_size 256 --batch_size 16 \
#             --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
#             --total_itrs 30000 \
#             2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import logging
from pathlib import Path
import datetime

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pprint

from knockknock import telegram_sender


"""
 ----------------------------- PASCAL VOC -----------------------------


# train base model from scratch
nohup sh -c 'python main.py \
    --gpu_id 1 \
    --random_seed 3 \
    --logdir logs/voc/base_seed_3 \
    --train_sample_ratio 0.75 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000' \
            2>&1 | tee -a nohup_outputs/voc/base.log &


# train from scratch with aug, ip2p
nohup sh -c 'python main.py \
    --gpu_id 2 \
    --random_seed 1 \
    --logdir logs/voc/seed_1_aug_ratio_0.15_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.75 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/pascal_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6.json \
    --aug_sample_ratio 0.15 \
        --model deeplabv3plus_mobilenet --dataset voc --year 2012 --crop_val \
        --lr 0.02 --crop_size 513 --batch_size 32 --output_stride 16 \
        --save_val_results --total_itrs 30000' \
        2>&1 | tee -a nohup_outputs/voc/output.log &
   
        
# aug jsons: 
    # first one, constant instuctions. Looks pretty good. basic, yet meaningful changes.
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_2x_constant_instructions_image_w_1.5_images_lpips_filter_0.1_0.6.json \
    # second one, instructions based on blip + gpt. Cool, but has much more artifacts than the first one.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/pascal_pascal_ip2p_2x_constant_instructions_image_w_1.5_with_blip_and_gpt_images_lpips_filter_0.1_0.6.json \
    # same with filter for small blobs (look above)
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_2x_constant_instructions_image_w_1.5_with_blip_and_gpt_images_masked__min_blob_size_100000_v0_lpips_filter_0.1_0.6.json \
    # third one, instructions based on both blip + gpt, and constant instructions. Looks pretty good. (provided the best reults yet)
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/pascal_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6.json \
    # forth one, instructions only constant instructions, 4 outputs. Probably not as good as the first one.
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_4x_image_w_1.5_only_constant_images_lpips_filter_0.1_0.6.json \
    # blip + gpt v1 (newer), 4 outputs. tbh, doesn't look good
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_pascal_ip2p_4x_image_w_1.5_blip_gpt_v1_images_masked__min_blob_size_100000__lpips_filter_0.1_0.6.json \
    # blip + gpt v2 (newer), 2 outputs. 
    --aug_json /mnt/raid/home/eyal_michaeli/git/DeepLabV3Plus-Pytorch/datasets/data/aug_json_files/pascal/ip2p/pascal_2023_0713_2225_51_pascal_ip2p_regular_2x_image_w_1.5_blip_gpt_v1_ratio_0.3_images_lpips_filter_0.1_0.6.json \

        
pkill -u eyal_michaeli tensorboard
nohup tensorboard --logdir=logs/voc --port=6006
nohup tensorboard --logdir=logs/cityscapes --port=6007
nohup tensorboard --logdir=logs/cityscapes_with_bug --port=6008


# send command after x minutes: (does it work??? YES): 
# nohup sh -c 'sleep 1m; ls' 2>&1 | tee -a bla.log &

----------------------------- CITYSCAPES -----------------------------

# train base model from scratch
nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_subset_2k_seed_1_base \
    --train_sample_ratio 0.67 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &


            
# using a subset (67%, 2k) for training data  
nohup sh -c 'python main.py \
    --gpu_id 0 \
    --random_seed 2 \
    --logdir logs/cityscapes/cs_base_run_seed_2_using_subset_2k \
    --train_sample_ratio 0.67 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/nohup_output-cs_base_run_seed_2_using_subset_2k.log &

            
# using extra test data   
nohup sh -c 'python main.py \
    --gpu_id 2 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_base_run_seed_1_extra_test_data_all_cities \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/nohup_output-cs_base_run_seed_1_extra_test_data_all_cities.log &

            
# resume training with diff seed
nohup sh -c 'python main.py \
    --gpu_id 0 \
    --random_seed 2 \
    --logdir logs/cityscapes/cs_base_run_continue \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results\
                  --continue_training --ckpt logs/2023_0520_2331_56_cs_base_run/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth' \
            2>&1 | tee -a nohup_outputs/nohup_output-.log &
                  

# train from scratch with aug, ip2p
nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_subset_2k_seed_1_aug_ratio_0.25_cs_ip2p_2x_constant_instructions_image_w_1.5_images_masked_person_rider_min_blob_size_100000_v0_lpips_filter_0.1_0.6 \
    --train_sample_ratio 0.67 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/cs_ip2p_2x_constant_instructions_image_w_1.5_images_masked_person_rider_min_blob_size_100000_v0_lpips_filter_0.1_0.6.json \
    --aug_sample_ratio 0.25 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/cs/nohup.log &

            
# train from scratch with aug, ip2p
nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_aug_ratio_0.1_cs_ip2p_2x_constant_instructions_image_w_1.5_no_double_aug \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/cs_ip2p_2x_constant_instructions_image_w_1.5/images.json \
    --aug_sample_ratio 0.1 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/nohup_output-cs_aug_ratio_0.1_cs_ip2p_2x_constant_instructions_image_w_1.5_no_double_aug.log &


# train from scratch with aug, MUNIT
nohup sh -c 'python main.py \
    --gpu_id 3 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_aug_run_munit_v0_cp_450k_style_std_1.85_lpips_filter_0.1_0.5_aug_ratio_0.15 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/2023_0603_1025_25_ampO1_lower_LR_lower_res_inference_inference_cp_000450000_style_std_1.85_lpips_filter_0.1_0.5.json \
    --aug_sample_ratio 0.15 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/nohup_output-cs_aug_run_munit_v0_cp_450k_style_std_1.85_lpips_filter_0.1_0.5_aug_ratio_0.15.log &

            
# train from scratch with aug, MUNIT
nohup sh -c 'python main.py \
    --gpu_id 0 \
    --random_seed 1 \
    --logdir logs/cityscapes/cs_subset_2k_aug_run_munit_style_recon_2_perceptual_1_style_1.5_aug_ratio_0.5 \
    --train_sample_ratio 0.67 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cs2cs-style_recon_2_perceptual_1/2023_0518_1805_39_ampO1_lower_LR/inference_cp_400k_style_std_1.5.json \
    --aug_sample_ratio 0.5 \
        --model deeplabv3plus_mobilenet --dataset cityscapes --lr 0.2  --crop_size 256 --batch_size 32 \
            --data_root /mnt/raid/home/eyal_michaeli/datasets/cityscapes --save_val_results' \
            2>&1 | tee -a nohup_outputs/cityscapes/nohup.log &
            
            
#### aug jsons ####

# ip2p
    # first try on cityscapes. ok results, but could be better
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/cs_ip2p_2x_constant_instructions_image_w_1.5_filtered_lpips_0.1_0.48.json \
    # same, filtered with lpips and min blob size

    # tried with magic brush, also ont really good. filtered with lpips 
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/cityscapes_2023_0708_2338_27_cityscapes_ip2p__2x_constant_instructions_image_w_1.5_images_lpips_filter_0.1_0.6.json
    # same, filtered with lpips and min blob size
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/2023_0708_2338_27_cityscapes_ip2p__2x_constant_instructions_image_w_1.5_images_masked_person_rider_min_blob_size_1562.5_v0_lpips_filter_0.1_0.6.json \

        
    # munit
    # v0
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cs2cs-style_recon_2_perceptual_1/2023_0518_1805_39_ampO1_lower_LR/inference_cp_400k_style_std_1.5.json \
    # v1
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/2023_0603_1025_25_ampO1_lower_LR_lower_res_inference_inference_cp_000450000_style_std_1.85_lpips_filter_0.1_0.5.json \
    # same, filtered with lpips and min blob size
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/2023_0603_1025_25_ampO1_lower_LR_lower_res_inference_inference_cp_000450000_style_std_1.85_masked_person_rider_min_blob_size_3125_v0_lpips_filter_0.1_0.5.json \


pkill -u eyal_michaeli tensorboard
tensorboard --logdir=logs/voc --port=6006
tensorboard --logdir=logs/cityscapes --port=6007

"""


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    # add arg to take only some amount for the train set
    parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                        help="ratio of train set to take")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=20e3,
                        help="max num of iters (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    
    parser.add_argument("--crop_size", type=int, default=256)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    
    parser.add_argument("--print_interval", type=int, default=50,
                        help="iterations interval of loss (default: 50)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="iterations interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Log directory
    parser.add_argument("--logdir", type=str, default=None,
                        help="path to the log directory")

    # augmentation options
    parser.add_argument("--aug_json", type=str, default=None,
                        help="path to augmentation json file")
    parser.add_argument("--aug_sample_ratio", type=float, default=0.5,
                        help="ratio to augment the original image")
    parser.add_argument("--limit_aug_per_image", type=int, default=None,
                        help="limit augmentations per image, default None, which is take all")
    return parser


def init_logging(logdir):
    r"""
    Create log directory for storing checkpoints and output images.
    Given a log dir like logs/test_run, creates a new directory logs/2020_0101_1234_test_run

    Args:
        logdir (str): Log directory name
    """
    # log dir
    date_uid = str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))
    logdir_path = Path(logdir)
    logdir = str(logdir_path.parent / f"{date_uid}_{logdir_path.name}")
    os.makedirs(logdir, exist_ok=True)
    # log file
    log_file = os.path.join(logdir, 'log.log')
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    return logdir


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                # et.ExtResize(size=opts.crop_size),
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtResize(size=opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform, 
                                    aug_json=opts.aug_json, aug_sample_ratio=opts.aug_sample_ratio, train_sample_ratio=opts.train_sample_ratio, limit_aug_per_image=opts.limit_aug_per_image)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtResize(size=opts.crop_size),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            #et.ExtRandomGaussianBlur(), if u want it, the default params are ok (I cheked)
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(size=opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root, split='train', transform=train_transform, aug_json=opts.aug_json, 
                               aug_sample_ratio=opts.aug_sample_ratio, train_sample_ratio=opts.train_sample_ratio, limit_aug_per_image=opts.limit_aug_per_image)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, epoch=0, iter=0):
    """Do validation and return specified samples"""
    metrics.reset()
    images_to_visualize = []
    if opts.save_val_results:
        if opts.logdir is not None:
            results_dir = os.path.join(opts.logdir, 'results')
            os.makedirs(results_dir, exist_ok=True)
        else:
            results_dir = 'results'
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    # divide results_dir into subdirs, that are named after the epoch
    results_dir = os.path.join(results_dir, str(epoch))
    os.makedirs(results_dir, exist_ok=True)
        
    with torch.no_grad():
        for batch_index, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if batch_index % 20 == 0 and opts.save_val_results:
                # if iter is higher than 5000: do it for iter % 5000 == 0
                # if iter is higher than 1000: do it for iter % 100 == 0
                # else: do it for iter % 200 == 0
                if iter > 5000:
                    if iter % 5000 != 0:
                        continue
                elif iter > 1000:
                    if iter % 1000 != 0:
                        continue
                else:
                    if iter % 200 != 0:
                        continue
                

                i = 0  # only save one image per batch, the first one

                images_to_visualize.append((
                    images[i].detach().cpu().numpy(), 
                    targets[i], 
                    preds[i]
                    ))
                # will images_to_visualize be the same every validation?
                # it should be, because the validation set is the same
                # but it is not, because the dataloader is shuffled

                image = images[i].detach().cpu().numpy()
                target = targets[i]
                pred = preds[i]

                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)
                pred = loader.dataset.decode_target(pred).astype(np.uint8)

                Image.fromarray(image).save(os.path.join(results_dir, f"{batch_index}_image_iter_{iter}.png"))
                Image.fromarray(target).save(os.path.join(results_dir, f"{batch_index}_gt_iter_{iter}.png"))
                Image.fromarray(pred).save(os.path.join(results_dir, f"{batch_index}_pred_iter_{iter}.png"))

                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig(os.path.join(results_dir, f"{batch_index}_overlay_iter_{iter}.png"), bbox_inches='tight', pad_inches=0.0)
                plt.close()

        score = metrics.get_results()

    return score, images_to_visualize


#@telegram_sender(token="2139102797:AAEieyZeXgDM9srZTsH2Aou7mSaUukfGQJA", chat_id=432012184)
def main(opts):
    
    # init logging
    if opts.logdir is not None:
        opts.logdir = init_logging(opts.logdir)

    
    writer = SummaryWriter(log_dir=str(Path(opts.logdir) / 'tb')) 
    
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # write the options to tensorboard, so that we know what we have been running
    # make sure it is readble
    text_for_tensorboard = pprint.pformat(str(vars(opts)))
    writer.add_text('opts', text_for_tensorboard)

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device(f'cuda')
    logging.info("CUDA_VISIBLE_DEVICES: %s" % opts.gpu_id)
    logging.info("Random Seed: %d \n" % opts.random_seed)
    logging.info("Options: %s" % pprint.pformat(vars(opts)))
    
    # print pid
    logging.info("PID: {}".format(os.getpid()))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    torch.cuda.manual_seed_all(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    # set deterministic cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    logging.info("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # compile with pytorch 2.0  (UPDATE: resulted in an error)
    # model = torch.compile(model)

    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        if opts.logdir is not None:
            path = os.path.join(opts.logdir, path)
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        logging.info("Model saved as %s" % path)

    if opts.logdir is not None:
        os.makedirs(opts.logdir, exist_ok=True)
        os.makedirs(os.path.join(opts.logdir, 'checkpoints'), exist_ok=True)

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            logging.info("Training state restored from %s" % opts.ckpt)
        logging.info("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        logging.info("[!] Training from scratch, no checkpoint file used")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, images_to_visualize = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        logging.info(metrics.to_str(val_score))
        return

    lr = optimizer.param_groups[0]['lr']
    if writer is not None:
        writer.add_scalar('learning_rate', lr, cur_itrs)
        
    interval_loss = 0
    while True: 
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        logging.info(f"\nCurrent epoch: {cur_epochs}")
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            # tensorboard
            if writer is not None:
                writer.add_scalar('Train_Loss', np_loss, cur_itrs)

            if (cur_itrs) % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval 
                logging.info("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0 or cur_itrs == opts.total_itrs or cur_itrs == 1:
                #save_ckpt(f"checkpoints/{opts.model}_{opts.dataset}_os{opts.output_stride}_iter{cur_itrs}_epoch{cur_epochs}_miou{best_score:.4f}.pth")
                logging.info("validation...")
                model.eval()
                val_score, images_to_visualize = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, epoch=cur_epochs, iter=cur_itrs)
                logging.info(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']

                    # convert the above line to f strinng
                    #save_ckpt(f'checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}_iter{cur_itrs}_epoch{cur_epochs}_miou{val_score["Mean IoU"]:.4f}.pth')
                    # instead of saving every best model, save only the last one
                    save_ckpt(f'checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}_miou{val_score["Mean IoU"]:.4f}.pth')

                # same for tensorboard
                if writer is not None:
                    writer.add_scalar('Val_Acc', val_score['Overall Acc'], cur_itrs)
                    writer.add_scalar('Val_mean_class_wise_acc', val_score['Mean Acc'], cur_itrs)
                    writer.add_scalar('Val_MIoU', val_score['Mean IoU'], cur_itrs)
                    # add also class IOU, but all in the same graph
                    # only if it's cityscapes
                    if opts.dataset in ['cityscapes', 'voc']:
                        for train_id, iou in val_score['Class IoU'].items():
                            if opts.dataset == 'cityscapes':
                                class_name = train_dst.train_id_to_class_name[train_id]
                            elif opts.dataset == 'voc':
                                class_name = train_dst.classes[train_id]
                            else:
                                raise NotImplementedError
                            
                            class_name = class_name.replace(' ', '_')
                            writer.add_scalar(f'Val_class_iou/{class_name}', iou, cur_itrs)
                    
                    # 1/4 the time do this
                    if cur_itrs % (opts.val_interval * 4) == 0:
                        for k, (img, target, lbl) in enumerate(images_to_visualize):
                            img = (denorm(img) * 255).astype(np.uint8)
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)
                            # write with a good title
                            writer.add_image(f"images/Val_image_pred{k}", concat_img, cur_itrs, dataformats='CHW')
                                         
                
                model.train()
            
            last_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != last_lr:
                writer.add_scalar('learning_rate', current_lr, cur_itrs)

            if cur_itrs >= opts.total_itrs:
                return


class Args:
    def __init__(self):
        # Dataset Options
        self.data_root = './datasets/data'
        self.dataset = 'voc'
        self.num_classes = None
        self.train_sample_ratio = 1.0

        # Deeplab Options
        self.model = 'deeplabv3plus_mobilenet'
        self.separable_conv = False
        self.output_stride = 16

        # Train Options
        self.test_only = False
        self.save_val_results = False
        self.total_itrs = 30e3
        self.lr = 0.01
        self.lr_policy = 'poly'
        self.step_size = 10000
        self.crop_val = False
        self.batch_size = 16
        self.val_batch_size = 4

        self.crop_size = 256

        self.ckpt = None
        self.continue_training = False

        self.loss_type = 'cross_entropy'
        self.gpu_id = 0
        self.weight_decay = 1e-4
        self.random_seed = 1

        self.print_interval = 50
        self.val_interval = 100
        self.download = False

        # PASCAL VOC Options
        self.year = '2012'

        # Log directory
        self.logdir = None

        # Augmentation options
        self.aug_json = None
        self.aug_sample_ratio = 0.5



if __name__ == '__main__':
    debug = False
    if debug:
        args = Args()
        args.gpu_id = 3
        args.random_seed = 3
        args.logdir = "logs/voc/seed_3_aug_ratio_0.3_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6"
        args.train_sample_ratio = 0.75
        args.aug_json = "/mnt/raid/home/eyal_michaeli/datasets/cityscapes/aug_json_files/cityscapes/ip2p/pascal_pascal_ip2p_2x_image_w_1.5_both_constant_instructions_with_blip_and_gpt_images_lpips_filter_0.1_0.6.json"
        args.aug_sample_ratio = 0.4
        args.model = "deeplabv3plus_mobilenet"
        args.dataset = "voc"
        args.year = "2012"
        args.crop_val = True
        args.lr = 0.02
        args.crop_size = 513
        args.batch_size = 32
        args.output_stride = 16
        args.save_val_results = True
        print("\n"*5, "running in DEBUG MODE".upper(), "\n"*5)

    else:
        args = get_argparser().parse_args()
    
    main(args)

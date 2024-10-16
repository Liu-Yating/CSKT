#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=5 \
python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--dataset_name $DATASET_NAME \
--loss_names 'sdm' \
--num_epoch 60 \
--root_dir '.../dataset_reid' \
--lr 3e-4 \
--depth 12 \
--n_ctx 4


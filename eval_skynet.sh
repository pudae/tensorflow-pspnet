#!/bin/bash

EVAL_DIR=./eval/skynet
DATASET_DIR=/home/pudae/dataset/ade20k/records
CHECKPOINT_PATH=./train/skynet/skynet_v1_50.ckpt

python eval_semantic_segmentation.py \
	--eval_dir=${EVAL_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--dataset_name=ade20k \
	--model_name=pspnet_v1_50 \
	--checkpoint_path=${CHECKPOINT_PATH} \
	--eval_image_size=224 \
	--classes=3 \
	--batch_size=12


#!/bin/bash

EVAL_DIR=./eval/pspnet
DATASET_DIR=/home/pudae/dataset/ade20k/records
CHECKPOINT_PATH=./train/pspnet/model.ckpt-168439

CUDA_VISIBLE_DEVICES=5 \
python eval_semantic_segmentation.py \
	--eval_dir=${EVAL_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--dataset_name=ade20k \
  --dataset_split_name=validation \
	--model_name=pspnet_v1_50 \
	--checkpoint_path=${CHECKPOINT_PATH} \
  --eval_image_size=473 \
  --batch_size=2


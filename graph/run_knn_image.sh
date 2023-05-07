#!/bin/sh
set -e
set -x

# build textual knn configs

## build KNN of text information without reranking operation

## NUSWide-Web
# --is_eval \
# --is_visualize \

# CUDA_VISIBLE_DEVICE=0 python3 visual_knn.py \
# --is_nuswide \
# --smooth_text \
# --model_name "gpt"

# CUDA_VISIBLE_DEVICE=0 python3 visual_knn_rerank.py \
# --is_nuswide \
# --smooth_text \
# --model_name "gpt"

## Google500

# CUDA_VISIBLE_DEVICE=1 python3 visual_knn.py \
# --is_visualize \
# --is_g500 \
# --is_eval \
# --smooth_text \
# --model_name "gpt"

# CUDA_VISIBLE_DEVICE=0 python3 visual_knn_rerank.py \
# --is_visualize \
# --is_g500 \
# --num_threads 8 \
# --is_eval \
# --smooth_text \
# --model_name "gpt"

## WebVision

# CUDA_VISIBLE_DEVICE=1 python3 visual_knn.py \
# --smooth_text \
# --model_name "gpt"

# --is_visualize \
# --is_eval

CUDA_VISIBLE_DEVICE=1 python3 visual_knn_rerank.py \
--smooth_text \
--model_name "gpt"

# --is_visualize \
# --is_eval

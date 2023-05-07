#!/bin/sh
set -e
set -x

# build textual knn configs

## build KNN of text information without reranking operation

## NUSWide-Web
###--is_visualize \
# --is_visualize \

CUDA_VISIBLE_DEVICE=0 python3 textual_knn.py \
--model_name "minilm" \
--is_nuswide \
--is_eval

CUDA_VISIBLE_DEVICE=0 python3 textual_knn.py \
--model_name "xlnet" \
--is_nuswide \
--is_eval

CUDA_VISIBLE_DEVICE=0 python3 textual_knn_rerank.py \
--model_name "minilm" \
--is_nuswide \
--is_eval

CUDA_VISIBLE_DEVICE=0 python3 textual_knn_rerank.py \
--model_name "xlnet" \
--is_nuswide \
--is_eval

## Google500

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn.py \
# --model_name "minilm" \
# --is_visualize \
# --is_g500

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn.py \
# --model_name "xlnet" \
# --is_visualize \
# --is_g500

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn_rerank.py \
# --model_name "minilm" \
# --is_visualize \
# --is_g500

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn_rerank.py \
# --model_name "xlnet" \
# --is_visualize \
# --is_g500

## WebVision

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn.py \
# --model_name "minilm" \
# --is_visualize

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn.py \
# --model_name "xlnet" \
# --is_visualize

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn_rerank.py \
# --model_name "minilm" \
# --is_visualize \
# --num_threads 8

# CUDA_VISIBLE_DEVICE=0 python3 textual_knn_rerank.py \
# --model_name "xlnet" \
# --is_visualize \
# --num_threads 8

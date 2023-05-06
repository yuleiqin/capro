#!/usr/bin/env bash
set -e
set -x

## GPU CONFIG
export HOST_GPU_NUM=1
export CUDA_VISIBLE_DEVICES=0
export DISTRIBUTED_BACKEND="nccl"
export NCCL_DEBUG_SUBSYS="INIT"
export NCCL_LAUNCH_MODE="GROUP"
export NCCL_P2P_DISABLE="0"
export NCCL_IB_DISABLE="1"
export NCCL_DEBUG="INFO"
export NCCL_SOCKET_IFNAME="eth1"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_SL="3"
export NCCL_NET_GDR_READ="1"
export NCCL_NET_GDR_LEVEL="3"
export NCCL_P2P_LEVEL="NVL"

## TRAINING CONFIG
export root_dir="dataset/webvision1k/tfrecord_webvision_train"
export pathlist="filelist/train_filelist_google_500_usable_tf_knn_rerank_smoothed_meta_minilm.txt"
# export pathlist="filelist/train_filelist_google_500_usable_tf_knn_rerank_smoothed_meta_gpt.txt"
# export pathlist="filelist/train_filelist_google_500_usable_tf_knn_rerank_smoothed_meta_xlnet.txt"

export root_dir_test_web="dataset/webvision1k/tfrecord_webvision_val"
export pathlist_test_web="filelist/val_webvision_1k_usable_tf_hi.txt"
export root_dir_test_target="dataset/imagenet/tfrecord_imgnet_val"
export pathlist_test_target="filelist/val_imagenet_1k_usable_tf.txt"

export root_dir_t=""
export pathlist_t=""
export N_CLASSES=500
export N_BATCHSIZE=256

export SAVE_DIR_ROOT="results"
export SAVE_DIR_NAME=web-g500-metapro-openset
export SAVE_DIR=${SAVE_DIR_ROOT}/${SAVE_DIR_NAME}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

export SAVE_DIR1=${SAVE_DIR}/bopro
export resume_path=results/web-g500-pretrained-metapro/bopro/checkpoint_best_imgnet.tar

if [ ! -d ${SAVE_DIR1} ]; then
    mkdir -p ${SAVE_DIR1}
fi

python3 -W ignore -u train.py --seed 1024 --use_meta_weights -1 --eval_only \
--root_dir ${root_dir} --pathlist_web ${pathlist} \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--arch "resnet50" --workers 16 --epochs 1 --frozen_encoder_epoch 0 \
--init-proto-epoch 1 --start_clean_epoch 1 --batch-size ${N_BATCHSIZE} \
--lr 1e-1 --cos --warmup_epoch 10 \
--exp-dir ${SAVE_DIR1} --start-epoch 0 --resume ${resume_path} \
--num-class ${N_CLASSES} --low-dim 128 \
--moco_queue 8192 --pseudo_th 0.6 \
--w-inst 1 --w-recn 1 --w-proto 1 \
--dist-url 'tcp://localhost:10354' --multiprocessing-distributed --world-size 1 --rank 0

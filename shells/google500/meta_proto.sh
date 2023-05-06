#!/usr/bin/env bash
set -e
set -x

## GPU CONFIG
export HOST_GPU_NUM=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
export pathlist_test_web="filelist/val_webvision_500_usable_tf_hi.txt"
export root_dir_test_target="dataset/imagenet/tfrecord_imgnet_val"
export pathlist_test_target="filelist/val_imagenet_500_usable_tf.txt"

export root_dir_t=""
export pathlist_t=""
export N_CLASSES=500
export N_BATCHSIZE=256

## SAVE CONFIG
export SAVE_DIR_ROOT="results"
export SAVE_DIR_NAME=web-g500-pretrained-metapro
export SAVE_DIR=${SAVE_DIR_ROOT}/${SAVE_DIR_NAME}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

export SAVE_DIR2=${SAVE_DIR}/bopro
if [ ! -d ${SAVE_DIR2} ]; then
    mkdir -p ${SAVE_DIR2}
fi

export resume_path=vanilla_ckpt/Google500_vanilla.tar

python3.9 -W ignore -u train.py --seed 1024 --use_meta_weights -1 --pseudo_fewshot \
--root_dir ${root_dir} --pathlist_web ${pathlist} \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--arch "resnet50" --workers 8 --epochs 120 --frozen_encoder_epoch 10 \
--init-proto-epoch 10 --start_clean_epoch 20 --batch-size ${N_BATCHSIZE} \
--topk 50 --lr 1e-1 --cos --warmup_epoch 10 --beta 0.1 \
--exp-dir ${SAVE_DIR2} --start-epoch 0 --resume ${resume_path} \
--num-class ${N_CLASSES} --low-dim 128 \
--moco_queue 8192 --pseudo_th 0.6 \
--w-inst 1 --w-recn 1 --w-proto 1 --update_proto_freq 1 \
--dist-url 'tcp://localhost:10233' --multiprocessing-distributed --world-size 1 --rank 0


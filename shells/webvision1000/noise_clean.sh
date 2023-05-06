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
export pathlist="filelist/train_filelist_webvision_1k_usable_tf_knn_rerank_smoothed_meta_minilm.txt"
# export pathlist="filelist/train_filelist_webvision_1k_usable_tf_knn_rerank_smoothed_meta_gpt.txt"
# export pathlist="filelist/train_filelist_webvision_1k_usable_tf_knn_rerank_smoothed_meta_xlnet.txt"

export root_dir_test_web="dataset/webvision1k/tfrecord_webvision_val"
export pathlist_test_web="filelist/val_webvision_1k_usable_tf_hi.txt"
export root_dir_test_target="dataset/imagenet/tfrecord_imgnet_val"
export pathlist_test_target="filelist/val_imagenet_1k_usable_tf.txt"

export root_dir_t=""
export pathlist_t=""
export N_CLASSES=1000
export N_BATCHSIZE=256

## SAVE CONFIG
export SAVE_DIR_ROOT="results"
export SAVE_DIR_NAME=web-webv1k-pretrained-metapro
export SAVE_DIR=${SAVE_DIR_ROOT}/${SAVE_DIR_NAME}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

export SAVE_DIR1=${SAVE_DIR}/bopro_ft
if [ ! -d ${SAVE_DIR1} ]; then
    mkdir -p ${SAVE_DIR1}
fi

export resume_path=${SAVE_DIR}/bopro/checkpoint_best_web.tar
export ANNO_PATH=${SAVE_DIR1}/pseudo_label.json

python3.9 -W ignore -u noise_cleaning.py --seed 1024 --use_meta_weights -1 --annotation ${ANNO_PATH} \
--root_dir ${root_dir} --pathlist_web ${pathlist} \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--arch "resnet50" --workers 16 --epochs 120 --frozen_encoder_epoch 10 \
--init-proto-epoch 10 --start_clean_epoch 20 --batch-size ${N_BATCHSIZE} \
--topk -50 --beta 0.1 --lr 1e-1 --cos --warmup_epoch 10 \
--exp-dir ${SAVE_DIR1} --start-epoch 0 --resume ${resume_path} \
--num-class ${N_CLASSES} --low-dim 128 \
--moco_queue 8192 --pseudo_th 0.6 \
--w-inst 1 --w-recn 1 --w-proto 1 \
--dist-url 'tcp://localhost:10253' --multiprocessing-distributed --world-size 1 --rank 0

### add mixup by "--mixup"
python3.9 -W ignore -u train.py --seed 1024 --use_meta_weights -1 --pseudo_fewshot --annotation ${ANNO_PATH} \
--root_dir ${root_dir} --pathlist_web ${pathlist} \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--arch "resnet50" --workers 16 --epochs 15 --mixup \
--frozen_encoder_epoch 16 --init-proto-epoch 16 --start_clean_epoch 16 \
--batch-size ${N_BATCHSIZE} --lr 1e-4 --cos --warmup_epoch 0 \
--exp-dir ${SAVE_DIR1} --start-epoch 0 --resume ${resume_path} \
--num-class ${N_CLASSES} --moco_queue 8192 \
--w-inst 0 --w-recn 0 --w-proto 0 \
--dist-url 'tcp://localhost:10263' --multiprocessing-distributed --world-size 1 --rank 0

import numpy as np
import os
from glob import glob
from knn_utils import global_build, sgc_precompute
import json
import sys
import argparse
from tqdm import tqdm
from offline_gnn import utils
sys.path.append("../")
from DataLoader.webvision_dataset import get_tfrecord_image
from rerank_feature import re_ranking
import multiprocessing as mp
from offline_gnn.utils import load_knn_graph


def load_meta_data(meta_data_json_path, tfrecord2img_idx_json_path=""):
    with open(meta_data_json_path, "r") as fr:
        meta_data_json = json.load(fr)
    if os.path.exists(tfrecord2img_idx_json_path):
        ## for nuswide datasets
        with open(tfrecord2img_idx_json_path, "r") as fr:
            tfrecord2img_idx = json.load(fr)
        img_idx2tfrecord = {v:k for k,v in tfrecord2img_idx.items()}
        meta_data_json_new = {}
        for k, v in meta_data_json.items():
            meta_data_json_new[img_idx2tfrecord[k]] = v 
        meta_data_json = meta_data_json_new
    return meta_data_json


def split_by_num_process(items_to_split, N_threads):
    """splits the items into N subsequences evenly"""
    num_each_split = len(items_to_split) // N_threads
    # remain_split = len(items_to_split) % N_threads
    nums_split = [[] for _ in range(N_threads)]
    for idx, item_to_split in enumerate(items_to_split):
        if num_each_split != 0:
            idx_split = idx // num_each_split
            if idx_split >= N_threads:
                idx_split = idx % N_threads
        else:
            idx_split = idx % N_threads
        nums_split[idx_split].append(item_to_split)
    nums_split = [num_split for num_split in nums_split if len(num_split) > 0]
    return nums_split


def rerank_knn(split_id, feat_ids_splits, knn_graph_knns, text_feature):
    rerank_knns = []
    feat_ids = feat_ids_splits[split_id]
    for feat_id in tqdm(feat_ids):
        ner, sim = knn_graph_knns[feat_id]
        probFeat = text_feature[feat_id:feat_id+1]
        galFeat = np.concatenate([text_feature[ner_id:ner_id+1] for ner_id in ner], axis=0)
        dist_rerank = re_ranking(probFeat, galFeat)[0]  # only one prob feature
        idx_rerank = np.argsort(dist_rerank)
        ner_rerank = [ner[idx] for idx in idx_rerank]
        # sim_rerank = [sim[idx] for idx in idx_rerank]
        sim_rerank = [dist_rerank[idx] for idx in idx_rerank]
        rerank_knns.append((feat_id, np.array(ner_rerank, dtype=np.int32), np.array(sim_rerank, dtype=np.float32)))
    return rerank_knns



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    build kNN of text features (smoothed) with rerank
    """
    parser.add_argument('--model_name', type=str, default="", help='which language model to encode textual info')
    parser.add_argument('--is_visualize', action='store_true', default=False, help='visualize the datasets')
    parser.add_argument('--is_g500', action='store_true', default=False, help='use webvision (Google500) dataset')
    parser.add_argument('--is_nuswide', action='store_true', default=False, help='use nuswide dataset')
    parser.add_argument('--is_debug', action='store_true', default=False, help='use debug mode')
    parser.add_argument('--num_threads', type=int, default=16, help='number of threads for multi-processing')
    parser.add_argument('--is_eval', action='store_true', default=False, help='use evaluation mode')

    args = parser.parse_args()

    is_visualize = args.is_visualize
    model_name = args.model_name
    is_nuswide = args.is_nuswide
    is_g500 = args.is_g500
    is_debug = args.is_debug
    N_threads = args.num_threads
    is_eval = args.is_eval

    ## read data from processed datasets 
    if is_nuswide:
        ## 处理nuswide数据集
        tfrecord_path = "dataset/nus_wide_web/tfrecord"
        if is_eval:
            root_path = "dataset/web_meta/nuswide_eval/text_graph_rerank"
            mapping_tfrecord2text_embd_idx_path = "dataset/web_meta/nuswide_eval/tfrecord2feat_id.json"
        else:
            root_path = "dataset/web_meta/nuswide/text_graph_rerank"
            # mapping_tfrecord2text_embd_idx_path = "dataset/nus_wide_web/meta_data_tf/mapping_tfrecord2embd_idx.txt"
            mapping_tfrecord2text_embd_idx_path = "dataset/web_meta/nuswide/tfrecord2feat_id.json"

        os.makedirs(root_path, exist_ok=True)
        if model_name == "minilm":
            if is_eval:
                filelist_path = "filelist/val_nus_81_tf.txt"
                text_feat_path = "dataset/web_meta/nuswide_eval/text_features/minilm/text_rerank_feats_eval.npy"
            else:
                filelist_path = "filelist/train_nus_81_tf_knn_rerank_smoothed_meta_minilm.txt"
                text_feat_path = "dataset/web_meta/nuswide/text_features/minilm/text_rerank_feats_train.npy"
        
        elif model_name == "xlnet":
            if is_eval:
                filelist_path = "filelist/val_nus_81_tf.txt"
                text_feat_path = "dataset/web_meta/nuswide_eval/text_features/minilm/text_rerank_feats_eval.npy"
            else:
                filelist_path = "filelist/train_nus_81_tf_knn_rerank_smoothed_meta_xlnet.txt"
                text_feat_path = "dataset/web_meta/nuswide/text_features/xlnet/text_rerank_feats_train.npy"
        
        text_all_json = "dataset/nus_wide_web/meta_data_tf/web_meta_by_img.json"
        tfrecord2img_idx_json_path = "dataset/nus_wide_web/meta_data_tf/tfrecord2imgidx.json"
    
    else:
        ## 处理WebVision数据集
        tfrecord_path = "dataset/webvision1k/tfrecord/tfrecord_webvision_train"
        root_path = "dataset/webvision1k/text_graph_rerank"
        os.makedirs(root_path, exist_ok=True)
        if is_g500:
            if model_name == "minilm":
                filelist_path = "filelist/low_resolution/train_filelist_google_500_usable_tf_knn_rerank_smoothed_meta_minilm.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_rerank_smoothed_meta_embd_minilm.npy"
            
            elif model_name == "xlnet":
                filelist_path = "filelist/low_resolution/train_filelist_google_500_usable_tf_knn_rerank_smoothed_meta_xlnet.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_rerank_smoothed_meta_embd_xlnet.npy"
        
        else:
            if model_name == "minilm":
                filelist_path = "filelist/low_resolution/train_filelist_webvision_1k_usable_tf_knn_rerank_smoothed_meta_minilm.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_rerank_smoothed_meta_embd_minilm.npy"
            
            elif model_name == "xlnet":
                filelist_path = "filelist/low_resolution/train_filelist_webvision_1k_usable_tf_knn_rerank_smoothed_meta_xlnet.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_rerank_smoothed_meta_embd_xlnet.npy"
        
        mapping_tfrecord2text_embd_idx_path = "dataset/webvision1k/meta_data_tf/mapping_tfrecord2embd_idx.txt"
        text_all_json = "dataset/webvision1k/meta_data_tf/meta_data.json"
        tfrecord2img_idx_json_path = ""

    print("processing mapping from tfrecord path to text embedding index")
    assert os.path.exists(mapping_tfrecord2text_embd_idx_path), "make sure {} path exists".format(mapping_tfrecord2text_embd_idx_path)
    if mapping_tfrecord2text_embd_idx_path.endswith(".txt"):
        mapping_tfrecord2text_embd_idx = utils.load_mapping_tfrecord2embd_idx(mapping_tfrecord2text_embd_idx_path)
    else:
        with open(mapping_tfrecord2text_embd_idx_path, "r") as fr:
            mapping_tfrecord2text_embd_idx = json.load(fr)

    assert os.path.exists(text_feat_path), "make sure {} path exists".format(text_feat_path)
    text_feature = np.load(text_feat_path)
    save_vis_path = os.path.join(root_path, "visualize_rerank_knn_nn_" + os.path.basename(text_feat_path).replace(".npy", ""))
    if (not is_nuswide) and is_g500:
        save_vis_path = os.path.join(root_path, "visualize_rerank_knn_nn_g500_" + os.path.basename(text_feat_path).replace(".npy", ""))
        valid_tfrecord_pathlist = []
        with open(filelist_path, "r") as fr:
            for line in fr.readlines():
                valid_tfrecord_path = line.strip().split(" ")[0]
                valid_tfrecord_pathlist.append(valid_tfrecord_path)
        text_feature_new = []
        mapping_tfrecord2text_embd_idx_new = {}
        feats_id2tfrecord = {}
        save_text_root = os.path.dirname(mapping_tfrecord2text_embd_idx_path)
        mapping_tfrecord2text_embd_idx_path_g500 = os.path.join(save_text_root,\
            "mapping_tfrecord2embd_idx_g500.txt")
        with open(mapping_tfrecord2text_embd_idx_path_g500, "w") as fw:
            for idx, valid_tfrecord_path in enumerate(valid_tfrecord_pathlist):
                fw.write("\t".join([str(valid_tfrecord_path), str(idx)]) + "\n")
                feats_id2tfrecord[idx] = valid_tfrecord_path
                mapping_tfrecord2text_embd_idx_new[valid_tfrecord_path] = idx
                feat_id = int(mapping_tfrecord2text_embd_idx[valid_tfrecord_path])
                text_feature_new.append(text_feature[feat_id:feat_id+1])
        text_feature = np.concatenate(text_feature_new, axis=0)
        mapping_tfrecord2text_embd_idx = mapping_tfrecord2text_embd_idx_new

    feats_id2tfrecord = {v:k for k,v in mapping_tfrecord2text_embd_idx.items()}
    print("text feature npy shape", text_feature.shape)
    if is_debug:
        text_feature = text_feature[:100]
        print("[debug] text feature npy shape", text_feature.shape)
    text_feature = text_feature / np.linalg.norm(text_feature, axis=1, keepdims=True)

    if is_nuswide:
        ## 4 x 10 ==> 40
        knn_graph = global_build(text_feature, dist_def="cosine:", k=40,\
            save_filename=os.path.join(root_path,
                "smoothed_text_{}_nuswide_train_knn".format(model_name)), build_graph_method='gpu')
        save_filename_rerank = save_filename=os.path.join(root_path,
            "rerank_smoothed_text_{}_nuswide_train_knn".format(model_name)) + '_k40.npz'
    else:
        ## 4 x 5 ==> 20
        if is_g500:
            knn_graph = global_build(text_feature, dist_def="cosine:", k=20,\
                save_filename=os.path.join(root_path,
                    "smoothed_text_{}_g500_train_knn".format(model_name)), build_graph_method='gpu')
            save_filename_rerank = save_filename=os.path.join(root_path,
                "rerank_smoothed_text_{}_g500_train_knn".format(model_name)) + '_k20.npz'
        else:
            knn_graph = global_build(text_feature, dist_def="cosine:", k=20,\
                save_filename=os.path.join(root_path,
                    "smoothed_text_{}_webvision_train_knn".format(model_name)), build_graph_method='gpu')
            save_filename_rerank = save_filename=os.path.join(root_path,
                "rerank_smoothed_text_{}_webvision_train_knn".format(model_name)) + '_k20.npz'

    knn_graph_knns = knn_graph.get_knns()
    if os.path.exists(save_filename_rerank):
        print('[faiss] read reranked knns from {}'.format(save_filename_rerank))
        # knn_graph_knns_rerank = [(knn[0, :].astype(np.int32), knn[1, :].astype(np.float32))
        #                 for knn in np.load(save_filename_rerank)['knns']]
        if is_nuswide:
            knn_graph_knns_rerank = load_knn_graph(save_filename_rerank, 40, False)
        else:
            knn_graph_knns_rerank = load_knn_graph(save_filename_rerank, 20, False)
    else:
        print('[rerank] start reranking knn-graphs')
        knn_graph_knns_rerank = [None for _ in range(len(knn_graph_knns))]
        knn_graph_ids = np.arange(len(knn_graph_knns))
        if N_threads <= 1:
            knn_graph_ids_splits = [knn_graph_ids]
            print("Start single-thread processing")
            rerank_knns = rerank_knn(0, knn_graph_ids_splits, knn_graph_knns, feats_all)
            for rerank_knn in rerank_knns:
                knn_graph_knns_rerank[rerank_knn[0]] = rerank_knn[1:]
            print("End single-thread processing")
        else:
            knn_graph_ids_splits = split_by_num_process(knn_graph_ids, N_threads)
            knn_graph_ids_splits = [knn_graph_ids_split for knn_graph_ids_split in knn_graph_ids_splits if len(knn_graph_ids_split) > 0]
            N_threads = min(N_threads, len(knn_graph_ids_splits))
            processes = []
            print("Start multiprocessing {} threads".format(N_threads))
            try:
                mp.set_start_method('spawn', force=True)
                print("Context MP spawned")
            except RuntimeError:
                pass
            pool = mp.Pool(processes=N_threads)
            for i in range(N_threads):
                processes.append(pool.apply_async(rerank_knn,\
                    args=(i, knn_graph_ids_splits, knn_graph_knns, text_feature)))
            pool.close()
            pool.join()
            for process in processes:
                rerank_knns = process.get()
                for rerank_knn in rerank_knns:
                    knn_graph_knns_rerank[rerank_knn[0]] = rerank_knn[1:]
            print("End multiprocessing")
        np.savez_compressed(save_filename_rerank, knns=knn_graph_knns_rerank)
        if is_nuswide:
            knn_graph_knns = load_knn_graph(knn_graph_knns, 40, False)
            knn_graph_knns_rerank = load_knn_graph(knn_graph_knns_rerank, 40, False)
        else:
            knn_graph_knns = load_knn_graph(knn_graph_knns, 20, False)
            knn_graph_knns_rerank = load_knn_graph(knn_graph_knns_rerank, 20, False)

    if is_nuswide:
        knn_graph_knns = [(knn_id[0][:10], knn_id[1][:10]) for knn_id in knn_graph_knns]
        knn_graph_knns_rerank = [(knn_id[0][:10], knn_id[1][:10]) for knn_id in knn_graph_knns_rerank]
    else:
        knn_graph_knns = [(knn_id[0][:5], knn_id[1][:5]) for knn_id in knn_graph_knns]
        knn_graph_knns_rerank = [(knn_id[0][:5], knn_id[1][:5]) for knn_id in knn_graph_knns_rerank]

    if is_visualize:
        assert os.path.exists(text_all_json), "make sure {} path exists".format(text_all_json)
        tfrecord2text = load_meta_data(text_all_json, tfrecord2img_idx_json_path)
        print("text info length", len(tfrecord2text))
        ## 按照特征序号从小到大
        for feat_id, knn_id in enumerate(knn_graph_knns_rerank[:20]):
            ner, sim = knn_id
            tfrecord_id = feats_id2tfrecord[feat_id].strip()
            if is_nuswide:
                text_id = tfrecord2text[tfrecord_id][:32]
            else:
                text_id = tfrecord2text[tfrecord_id]["text"][:32]
            text_id = text_id.replace(" ", "_").replace(",", "_").replace("/", "_")

            save_tfrecord_idx = str(feat_id) + "_" + tfrecord_id
            save_vis_tfrecord_path = os.path.join(save_vis_path, save_tfrecord_idx)
            os.makedirs(save_vis_tfrecord_path, exist_ok=True)

            tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
            img_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
            save_img_id_path = os.path.join(save_vis_tfrecord_path, "img_{}.jpg".format(text_id))
            img_id.save(save_img_id_path)

            for ner_idx, sim_idx in zip(ner, sim):
                tfrecord_id = feats_id2tfrecord[ner_idx].strip()
                if is_nuswide:
                    text_id = tfrecord2text[tfrecord_id][:32]
                else:
                    text_id = tfrecord2text[tfrecord_id]["text"][:32]
                text_id = text_id.replace(" ", "_").replace(",", "_").replace("/", "_")

                tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
                img_ner_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
                save_img_ner_id_path = os.path.join(save_vis_tfrecord_path,\
                    "img_neighbor_{}_{}_{}.jpg".format(ner_idx, sim_idx, text_id))
                img_ner_id.save(save_img_ner_id_path)
        
        ## 按照不一致程度从小到大
        ners_set = [set(knn_id[0].tolist()) for knn_id in knn_graph_knns]
        ners_set_rerank = [set(knn_id[0].tolist()) for knn_id in knn_graph_knns_rerank]
        knn_graph_knns_intersect = [len(ner.intersection(ner_rerank)) for ner, ner_rerank in zip(ners_set, ners_set_rerank)]
        knn_diff_maximum_ids = np.argsort(np.array(knn_graph_knns_intersect))

        if is_g500:
            save_vis_path = os.path.join(root_path, "visualize_rerank_knn_nn_g500_sorted_" + os.path.basename(text_feat_path).replace(".npy", ""))
        else:
            save_vis_path = os.path.join(root_path, "visualize_rerank_knn_nn_sorted_" + os.path.basename(text_feat_path).replace(".npy", ""))
        for feat_id in knn_diff_maximum_ids[:20]:
            tfrecord_id = feats_id2tfrecord[feat_id].strip()

            if is_nuswide:
                text_id = tfrecord2text[tfrecord_id][:32]
            else:
                text_id = tfrecord2text[tfrecord_id]["text"][:32]
            text_id = text_id.replace(" ", "_").replace(",", "_").replace("/", "_")

            save_tfrecord_idx = str(feat_id) + "_" + tfrecord_id
            save_vis_tfrecord_path = os.path.join(save_vis_path, save_tfrecord_idx)
            os.makedirs(save_vis_tfrecord_path, exist_ok=True)

            tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
            img_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
            save_img_id_path = os.path.join(save_vis_tfrecord_path, "img_{}.jpg".format(text_id))
            img_id.save(save_img_id_path)

            ner, sim = knn_graph_knns[feat_id]
            save_vis_tfrecord_path_org = os.path.join(save_vis_tfrecord_path, "original")
            os.makedirs(save_vis_tfrecord_path_org, exist_ok=True)
            for ner_idx, sim_idx in zip(ner, sim):
                tfrecord_id = feats_id2tfrecord[ner_idx].strip()

                if is_nuswide:
                    text_id = tfrecord2text[tfrecord_id][:32]
                else:
                    text_id = tfrecord2text[tfrecord_id]["text"][:32]
                text_id = text_id.replace(" ", "_").replace(",", "_").replace("/", "_")

                tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
                img_ner_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
                save_img_ner_id_path = os.path.join(save_vis_tfrecord_path_org,\
                    "img_neighbor_{}_{}_{}.jpg".format(ner_idx, sim_idx, text_id))
                img_ner_id.save(save_img_ner_id_path)
            
            ner_rerank, sim_rerank = knn_graph_knns_rerank[feat_id]
            save_vis_tfrecord_path_rerank = os.path.join(save_vis_tfrecord_path, "rerank")
            os.makedirs(save_vis_tfrecord_path_rerank, exist_ok=True)
            for ner_idx, sim_idx in zip(ner_rerank, sim_rerank):
                tfrecord_id = feats_id2tfrecord[ner_idx].strip()

                if is_nuswide:
                    text_id = tfrecord2text[tfrecord_id][:32]
                else:
                    text_id = tfrecord2text[tfrecord_id]["text"][:32]
                text_id = text_id.replace(" ", "_").replace(",", "_").replace("/", "_")

                tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
                img_ner_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
                save_img_ner_id_path = os.path.join(save_vis_tfrecord_path_rerank,\
                    "img_neighbor_{}_{}_{}.jpg".format(ner_idx, sim_idx, text_id))
                img_ner_id.save(save_img_ner_id_path)


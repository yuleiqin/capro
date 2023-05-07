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
    build kNN of visual features (smoothed) and smooth text features with rerank
    """
    parser.add_argument('--model_name', type=str, default="", help='which language model to encode textual info')
    parser.add_argument('--is_visualize', action='store_true', default=False, help='visualize the datasets')
    parser.add_argument('--is_g500', action='store_true', default=False, help='use webvision (Google500) dataset')
    parser.add_argument('--is_nuswide', action='store_true', default=False, help='use nuswide dataset')
    parser.add_argument('--smooth_text', action='store_true', default=False, help='smooth text features')
    parser.add_argument('--is_debug', action='store_true', default=False, help='use debug mode')
    parser.add_argument('--num_threads', type=int, default=16, help='number of threads for multi-processing')
    parser.add_argument('--is_eval', action='store_true', default=False, help='use evaluation mode')

    args = parser.parse_args()

    is_visualize = args.is_visualize
    model_name = args.model_name
    is_nuswide = args.is_nuswide
    is_g500 = args.is_g500
    is_debug = args.is_debug
    smooth_text = args.smooth_text
    N_threads = args.num_threads

    is_eval = args.is_eval

    if is_nuswide:
        tfrecord_path = "dataset/nus_wide_web/tfrecord"
        if is_eval:
            save_root = "dataset/web_meta/nuswide_eval/image_graph_rerank"
            tfrecord2feats_all_id_path = "dataset/web_meta/nuswide_eval/tfrecord2feat_id.json"
            root_path = "results/web-nuswide-pretrained-feat_save_FULL/stage1_test"
            filelist_path = "filelist/val_nus_81_tf.txt"
        else:
            # feats_all_path = "dataset/web_meta/nuswide/image_features/image_feats_train.npy"
            save_root = "dataset/web_meta/nuswide/image_graph"
            # tfrecord2feats_all_id_path = "dataset/web_meta/nuswide/tfrecord2feat_id.json"
            tfrecord2feats_all_id_path = "dataset/web_meta/nuswide/tfrecord2feat_id.json"
            root_path = "results/web-nuswide-pretrained-feat_save_FULL/stage1"
            filelist_path = "filelist/train_nus_81_tf.txt"
    else:

        if is_eval:
            tfrecord_path = "dataset/webvision1k/tfrecord/tfrecord_webvision_val_hd"
            if is_g500:
                root_path = "results/web-g500-pretrained-feat_save_FULL/stage1/WebVision"
                save_root = "dataset/web_meta/google500_eval/image_graph_rerank"
                filelist_path = "filelist/val_webvision_500_usable_tf_hi.txt"
            else:
                root_path = "results/web-webv1k_eval-pretrained-feat_save_FULL/stage1/WebVision"
                save_root = "dataset/web_meta/webvision_eval/image_graph_rerank"
                filelist_path = "filelist/val_webvision_1k_usable_tf_hi.txt"
        else:
            tfrecord_path = "dataset/webvision1k/tfrecord/tfrecord_webvision_train"
            root_path = "results/web-webv1k-pretrained-feat_save_FULL/stage1"
            # save_root = "dataset/webvision1k/image_graph_rerank"
            save_root = "dataset/web_meta/webvision/image_graph_rerank"
            if is_g500:
                filelist_path = "filelist/low_resolution/train_filelist_google_500_usable_tf.txt"
            else:
                filelist_path = "filelist/low_resolution/train_filelist_webvision_1k_usable_tf.txt"
        tfrecord2feats_all_id_path = os.path.join(root_path, "tfrecord2feats_all_id.json")

    feats_all_path = os.path.join(root_path, "feats_all.npy")
    # feats_all_path = "dataset/web_meta/nuswide/image_features/image_feats_train.npy"
    # tfrecord2feats_all_id_path = os.path.join(root_path, "tfrecord2feats_all_id.json")

    if not (os.path.exists(feats_all_path) and os.path.exists(tfrecord2feats_all_id_path)):
        ### 将数据集整理成有效形式
        feats_pathlist = list(glob(os.path.join(root_path, "*save_feats*.npy")))
        feats_pathlist = sorted(feats_pathlist, key=lambda x:int(os.path.basename(x).split(".npy")[0].split("_")[-1]))
        feats_all = []
        tfrecord_all = []

        for feat_path in feats_pathlist:
            feat = np.load(feat_path)
            feats_all.append(feat)
            txt_path = os.path.join(os.path.dirname(feat_path),\
                os.path.basename(feat_path).replace("save_feats", "tfrecord_names_all").replace(".npy", ".txt"))
            assert(os.path.exists(txt_path))
            with open(txt_path, "r") as fr:
                txt_lines = fr.readlines()
                assert(len(txt_lines) == len(feat))
                for txt_line in txt_lines:
                    tfrecord_name = os.path.basename(txt_line.strip())
                    tfrecord_all.append(tfrecord_name)
        
        feats_all = np.concatenate(feats_all, axis=0)
        if os.path.exists(tfrecord2feats_all_id_path):
            with open(tfrecord2feats_all_id_path, "r") as fr:
                tfrecord2feats_id = json.load(fr)
            idx_resorted = []
            for tfrecord_name in tfrecord_all:
                idx_resorted.append(int(tfrecord2feats_id[tfrecord_name]))
            idx_resorted = np.array(idx_resorted)
            print("index resorted", idx_resorted)
            assert(len(idx_resorted) == len(feats_all))
            feats_all = feats_all[idx_resorted]
        else:
            tfrecord2feats_id = {}
            for idx, tfrecord_name in enumerate(tfrecord_all):
                tfrecord2feats_id[tfrecord_name] = idx
            with open(tfrecord2feats_all_id_path, "w") as fw:
                json.dump(tfrecord2feats_id, fw)
        np.save(feats_all_path, feats_all)

    else:
        with open(tfrecord2feats_all_id_path, "r") as fr:
            tfrecord2feats_id = json.load(fr)
        feats_all = np.load(feats_all_path)
    feats_all /= np.linalg.norm(feats_all, axis=1, keepdims=True)

    # if (not is_nuswide) and is_g500:
    #     valid_tfrecord_pathlist = []
    #     with open(filelist_path, "r") as fr:
    #         for line in fr.readlines():
    #             valid_tfrecord_path = line.strip().split(" ")[0]
    #             valid_tfrecord_pathlist.append(valid_tfrecord_path)
    #     print("[google500] number of tfrecord pathlist = {}".format(len(valid_tfrecord_pathlist)))
    #     feats_all_new = []
    #     tfrecord2feats_all_id_path = os.path.join(root_path, "tfrecord2feats_all_id_g500.json")
    #     tfrecord2feats_id_new = {}

    #     for idx, valid_tfrecord_path in enumerate(valid_tfrecord_pathlist):
    #         tfrecord2feats_id_new[valid_tfrecord_path] = idx
    #         feat_id = int(tfrecord2feats_id[valid_tfrecord_path])
    #         feats_all_new.append(feats_all[feat_id:feat_id+1])

    #     with open(tfrecord2feats_all_id_path, "w") as fw:
    #         json.dump(tfrecord2feats_id_new, fw)

    #     feats_all_new = np.concatenate(feats_all_new, axis=0)
    #     feats_all = feats_all_new
    #     tfrecord2feats_id = tfrecord2feats_id_new

    print("feats all shape", feats_all.shape)
    print("tfrecord2feats_id shape", len(tfrecord2feats_id))
    if is_debug:
        feats_all = feats_all[:10]
        print("[debug] feats all shape", feats_all.shape)
    feats_id2tfrecord = {int(v):k for k,v in tfrecord2feats_id.items()}

    if is_nuswide:
        knn_graph = global_build(feats_all, dist_def="cosine:", k=40,\
            save_filename=os.path.join(save_root, "nuswide_train_knn"), build_graph_method='gpu')
        save_vis_path = os.path.join(save_root, "rerank_image_visualize_knn_nn_nuswide")
        save_filename_rerank = save_filename=os.path.join(save_root, "rerank_nuswide_train_knn") + '_k40.npz'

    else:
        if is_g500:
            knn_graph = global_build(feats_all, dist_def="cosine:", k=20,\
                save_filename=os.path.join(save_root, "google500_train_knn"), build_graph_method='gpu')
            save_vis_path = os.path.join(save_root, "rerank_image_visualize_knn_nn_google500")
            save_filename_rerank = save_filename=os.path.join(save_root, "rerank_google500_train_knn") + '_k20.npz'

        else:
            knn_graph = global_build(feats_all, dist_def="cosine:", k=20,\
                save_filename=os.path.join(save_root, "webvision_train_knn"), build_graph_method='gpu')
            save_vis_path = os.path.join(save_root, "rerank_image_visualize_knn_nn_webvision")
            save_filename_rerank = save_filename=os.path.join(save_root, "rerank_webvision_train_knn") + '_k20.npz'
    os.makedirs(save_vis_path, exist_ok=True)

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
                    args=(i, knn_graph_ids_splits, knn_graph_knns, feats_all)))
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

    knn_graph.knns = knn_graph_knns_rerank
    print("KNN Loaded")
    if is_visualize:
        ## 按照特征序号从小到大
        for feat_id, knn_id in enumerate(knn_graph_knns_rerank[:20]):
            ner, sim = knn_id
            tfrecord_id = feats_id2tfrecord[feat_id].strip()
            save_tfrecord_idx = str(feat_id) + "_" + tfrecord_id
            save_vis_tfrecord_path = os.path.join(save_vis_path, save_tfrecord_idx)
            os.makedirs(save_vis_tfrecord_path, exist_ok=True)

            tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
            img_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
            save_img_id_path = os.path.join(save_vis_tfrecord_path, "img.jpg")
            img_id.save(save_img_id_path)
            for ner_idx, sim_idx in zip(ner, sim):
                tfrecord_id = feats_id2tfrecord[ner_idx].strip()
                tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
                img_ner_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
                save_img_ner_id_path = os.path.join(save_vis_tfrecord_path,\
                    "img_neighbor_{}_{}.jpg".format(ner_idx, sim_idx))
                img_ner_id.save(save_img_ner_id_path)
        
        ## 按照不一致程度从小到大
        ners_set = [set(knn_id[0].tolist()) for knn_id in knn_graph_knns]
        ners_set_rerank = [set(knn_id[0].tolist()) for knn_id in knn_graph_knns_rerank]
        knn_graph_knns_intersect = [len(ner.intersection(ner_rerank)) for ner, ner_rerank in zip(ners_set, ners_set_rerank)]
        knn_diff_maximum_ids = np.argsort(np.array(knn_graph_knns_intersect))

        if is_nuswide:
            save_vis_path = os.path.join(save_root, "rerank_image_visualize_knn_nn_nuswide_sorted")
        else:
            if is_g500:
                save_vis_path = os.path.join(save_root, "rerank_image_visualize_knn_nn_google500_sorted")
            else:
                save_vis_path = os.path.join(save_root, "rerank_image_visualize_knn_nn_webvision_sorted")
        os.makedirs(save_vis_path, exist_ok=True)

        for feat_id in knn_diff_maximum_ids[:20]:
            tfrecord_id = feats_id2tfrecord[feat_id].strip()
            save_tfrecord_idx = str(feat_id) + "_" + tfrecord_id
            save_vis_tfrecord_path = os.path.join(save_vis_path, save_tfrecord_idx)
            os.makedirs(save_vis_tfrecord_path, exist_ok=True)

            tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
            img_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
            save_img_id_path = os.path.join(save_vis_tfrecord_path, "img.jpg")
            img_id.save(save_img_id_path)

            ner, sim = knn_graph_knns[feat_id]
            save_vis_tfrecord_path_org = os.path.join(save_vis_tfrecord_path, "original")
            os.makedirs(save_vis_tfrecord_path_org, exist_ok=True)
            for ner_idx, sim_idx in zip(ner, sim):
                tfrecord_id = feats_id2tfrecord[ner_idx].strip()
                tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
                img_ner_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
                save_img_ner_id_path = os.path.join(save_vis_tfrecord_path_org,\
                    "img_neighbor_{}_{}.jpg".format(ner_idx, sim_idx))
                img_ner_id.save(save_img_ner_id_path)
            
            ner_rerank, sim_rerank = knn_graph_knns_rerank[feat_id]
            save_vis_tfrecord_path_rerank = os.path.join(save_vis_tfrecord_path, "rerank")
            os.makedirs(save_vis_tfrecord_path_rerank, exist_ok=True)
            for ner_idx, sim_idx in zip(ner_rerank, sim_rerank):
                tfrecord_id = feats_id2tfrecord[ner_idx].strip()
                tfrecord_name, tfrecord_offset = tfrecord_id.split("@")
                img_ner_id = get_tfrecord_image(os.path.join(tfrecord_path, tfrecord_name), int(tfrecord_offset))
                save_img_ner_id_path = os.path.join(save_vis_tfrecord_path_rerank,\
                    "img_neighbor_{}_{}.jpg".format(ner_idx, sim_idx))
                img_ner_id.save(save_img_ner_id_path)

    if smooth_text:
        ## whether to smooth text embeddings
        if is_nuswide:
            if is_eval:
                if model_name == "minilm":
                    meta_info_path = "dataset/web_meta/nuswide_eval/text_features/minilm/text_feats_eval_raw.npy"
                elif model_name == "xlnet":
                    meta_info_path = "dataset/web_meta/nuswide_eval/text_features/xlnet/text_feats_eval_raw.npy"
                assert os.path.exists(meta_info_path), "{} exists".format(meta_info_path)
            else:
                if model_name == "minilm":
                    meta_info_path = "dataset/nus_wide_web/meta_data_tf/meta_embd_minilm.npy"
                elif model_name == "xlnet":
                    meta_info_path = "dataset/nus_wide_web/meta_data_tf/meta_embd_xlnet.npy"
                elif model_name == "gpt":
                    meta_info_path = "dataset/nus_wide_web/meta_data_tf/meta_embd_gpt.npy"
                assert(os.path.exists(meta_info_path)), "{} exists".format(meta_info_path)
                tfrecord2img_idx_path = "dataset/nus_wide_web/meta_data_tf/tfrecord2imgidx.json"
                with open(tfrecord2img_idx_path, "r") as fr:
                    tfrecord2img_idx = json.load(fr)

                img_idx2meta_idx_path = "dataset/nus_wide_web/meta_data_tf/mapping_img_idx2embd_idx.txt"
                img_idx2meta_idx = {}
                with open(img_idx2meta_idx_path, "r") as fr:
                    for line in fr.readlines():
                        img_idx, meta_idx = line.strip().split("\t")
                        img_idx2meta_idx[img_idx] = int(meta_idx)

        else:
            if model_name == "minilm":
                meta_info_path = "dataset/webvision1k/meta_data_tf/meta_embd_minilm.npy"
            elif model_name == "xlnet":
                meta_info_path = "dataset/webvision1k/meta_data_tf/meta_embd_xlnet.npy"
            elif model_name == "gpt":
                meta_info_path = "dataset/webvision1k/meta_data_tf/meta_embd_gpt.npy"
            assert(os.path.exists(meta_info_path)), "{} exists".format(meta_info_path)
            tfrecord2meta_idx_path = "dataset/webvision1k/meta_data_tf/mapping_tfrecord2index.txt"
            tfrecord2meta_idx = {}
            with open(tfrecord2meta_idx_path, "r") as fr:
                for line in fr.readlines():
                    img_idx, tfrecord = line.strip().split("\t")
                    tfrecord2meta_idx[tfrecord] = int(img_idx)

        print("Start loading text features")
        if is_eval and is_nuswide:
            save_meta_info_path = meta_info_path
        else:
            save_meta_info_path = os.path.join(root_path, "text_all_" + os.path.basename(meta_info_path))

        if not (os.path.exists(save_meta_info_path)):
            # ### 按照训练集的样本进行重新组织
            meta_info_all = np.load(meta_info_path)
            print("meta_info_all shape", meta_info_all.shape)
            text_feature = []

            for feat_id in range(len(feats_id2tfrecord)):
                tfrecord_name_id = feats_id2tfrecord[feat_id].strip()
                if is_nuswide:
                    meta_id = img_idx2meta_idx[tfrecord2img_idx[tfrecord_name_id]]
                else:
                    meta_id = tfrecord2meta_idx[tfrecord_name_id]
                text_feature.append(meta_info_all[meta_id])
            
            text_feature = np.array(text_feature)
            np.save(save_meta_info_path, text_feature)
        else:
            text_feature = np.load(save_meta_info_path)
        print("text feature npy shape", text_feature.shape)

        print("Start smooth text features")
        text_feature = sgc_precompute(text_feature, knn_graph, self_weight=0,\
            edge_weight=True, degree=1)
        np.save(os.path.join(root_path, "knn_rerank_smoothed_" + os.path.basename(meta_info_path)), text_feature)


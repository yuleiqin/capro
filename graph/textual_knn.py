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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    build kNN of text features (smoothed)
    """
    parser.add_argument('--model_name', type=str, default="", help='which language model to encode textual info')
    parser.add_argument('--is_visualize', action='store_true', default=False, help='visualize the datasets')
    parser.add_argument('--is_g500', action='store_true', default=False, help='use webvision (Google500) dataset')
    parser.add_argument('--is_nuswide', action='store_true', default=False, help='use nuswide dataset')
    parser.add_argument('--is_debug', action='store_true', default=False, help='use debug mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='use evaluation mode')

    args = parser.parse_args()

    is_visualize = args.is_visualize
    model_name = args.model_name
    is_nuswide = args.is_nuswide
    is_g500 = args.is_g500
    is_debug = args.is_debug
    is_eval = args.is_eval

    ## read data from processed datasets 
    if is_nuswide:
        ## 处理nuswide数据集
        tfrecord_path = "dataset/nus_wide_web/tfrecord"
        if is_eval:
            root_path = "dataset/web_meta/nuswide_eval/text_graph"
            mapping_tfrecord2text_embd_idx_path = "dataset/web_meta/nuswide_eval/tfrecord2feat_id.json"
        else:
            # root_path = "dataset/nus_wide_web/text_graph"
            root_path = "dataset/web_meta/nuswide/text_graph"
            # mapping_tfrecord2text_embd_idx_path = "dataset/nus_wide_web/meta_data_tf/mapping_tfrecord2embd_idx.txt"
            mapping_tfrecord2text_embd_idx_path = "dataset/web_meta/nuswide/tfrecord2feat_id.json"

        os.makedirs(root_path, exist_ok=True)
        if model_name == "minilm":
            if is_eval:
                filelist_path = "filelist/val_nus_81_tf.txt"
                text_feat_path = "dataset/web_meta/nuswide_eval/text_features/minilm/text_feats_eval.npy"
            else:
                filelist_path = "filelist/train_nus_81_tf_knn_rerank_smoothed_meta_minilm.txt"
                # text_feat_path = "dataset/nus_wide_web/meta_data_tf/knn_smoothed_meta_embd_minilm.npy"
                text_feat_path = "dataset/web_meta/nuswide/text_features/minilm/text_feats_train.npy"

        elif model_name == "xlnet":
            if is_eval:
                filelist_path = "filelist/val_nus_81_tf.txt"
                text_feat_path = "dataset/web_meta/nuswide_eval/text_features/xlnet/text_feats_eval.npy"
            else:
                filelist_path = "filelist/train_nus_81_tf_knn_rerank_smoothed_meta_xlnet.txt"
                # text_feat_path = "dataset/nus_wide_web/meta_data_tf/knn_smoothed_meta_embd_xlnet.npy"
                text_feat_path = "dataset/web_meta/nuswide/text_features/xlnet/text_feats_train.npy"
        text_all_json = "dataset/nus_wide_web/meta_data_tf/web_meta_by_img.json"
        tfrecord2img_idx_json_path = "dataset/nus_wide_web/meta_data_tf/tfrecord2imgidx.json"
    
    else:
        ## 处理WebVision数据集
        tfrecord_path = "dataset/webvision1k/tfrecord/tfrecord_webvision_train"
        root_path = "dataset/webvision1k/text_graph"
        os.makedirs(root_path, exist_ok=True)
        if is_g500:
            if model_name == "minilm":
                filelist_path = "filelist/low_resolution/train_filelist_google_500_usable_tf_knn_rerank_smoothed_meta_minilm.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_smoothed_meta_embd_minilm.npy"
            elif model_name == "xlnet":
                filelist_path = "filelist/low_resolution/train_filelist_google_500_usable_tf_knn_rerank_smoothed_meta_xlnet.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_smoothed_meta_embd_xlnet.npy"
        else:
            if model_name == "minilm":
                filelist_path = "filelist/low_resolution/train_filelist_webvision_1k_usable_tf_knn_rerank_smoothed_meta_minilm.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_smoothed_meta_embd_minilm.npy"
            elif model_name == "xlnet":
                filelist_path = "filelist/low_resolution/train_filelist_webvision_1k_usable_tf_knn_rerank_smoothed_meta_xlnet.txt"
                text_feat_path = "dataset/webvision1k/meta_data_tf/knn_smoothed_meta_embd_xlnet.npy"

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
    save_vis_path = os.path.join(root_path, "visualize_knn_nn_" + os.path.basename(text_feat_path).replace(".npy", ""))
    if (not is_nuswide) and is_g500:
        save_vis_path = os.path.join(root_path, "visualize_knn_nn_g500_" + os.path.basename(text_feat_path).replace(".npy", ""))
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
        text_feature = text_feature[:10]
        print("[debug] text feature npy shape", text_feature.shape)
    text_feature = text_feature / np.linalg.norm(text_feature, axis=1, keepdims=True)
    if is_nuswide:
        knn_graph = global_build(text_feature, dist_def="cosine:", k=10,\
            save_filename=os.path.join(root_path,
                "smoothed_text_{}_nuswide_train_knn".format(model_name)), build_graph_method='gpu')
    else:
        if is_g500:
            knn_graph = global_build(text_feature, dist_def="cosine:", k=5,\
                save_filename=os.path.join(root_path,
                    "smoothed_text_{}_g500_train_knn".format(model_name)), build_graph_method='gpu')
        else:
            knn_graph = global_build(text_feature, dist_def="cosine:", k=5,\
                save_filename=os.path.join(root_path,
                    "smoothed_text_{}_webvision_train_knn".format(model_name)), build_graph_method='gpu')

    if is_visualize:
        assert os.path.exists(text_all_json), "make sure {} path exists".format(text_all_json)
        tfrecord2text = load_meta_data(text_all_json, tfrecord2img_idx_json_path)
        print("text info length", len(tfrecord2text))
        for feat_id, knn_id in enumerate(knn_graph.knns[:20]):
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

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
    build kNN of visual features (smoothed) and smooth text features
    """
    parser.add_argument('--model_name', type=str, default="", help='which language model to encode textual info')
    parser.add_argument('--is_visualize', action='store_true', default=False, help='visualize the datasets')
    parser.add_argument('--is_g500', action='store_true', default=False, help='use webvision (Google500) dataset')
    parser.add_argument('--is_nuswide', action='store_true', default=False, help='use nuswide dataset')
    parser.add_argument('--smooth_text', action='store_true', default=False, help='smooth text features')
    parser.add_argument('--is_debug', action='store_true', default=False, help='use debug mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='use evaluation mode')

    args = parser.parse_args()

    is_visualize = args.is_visualize
    model_name = args.model_name
    is_nuswide = args.is_nuswide
    is_g500 = args.is_g500
    is_debug = args.is_debug
    smooth_text = args.smooth_text
    is_eval = args.is_eval

    if is_nuswide:
        tfrecord_path = "dataset/nus_wide_web/tfrecord"
        if is_eval:
            save_root = "dataset/web_meta/nuswide_eval/image_graph"
            # tfrecord2feats_all_id_path = "dataset/web_meta/nuswide_eval/tfrecord2feat_id.json"
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
                save_root = "dataset/web_meta/google500_eval/image_graph"
                filelist_path = "filelist/val_webvision_500_usable_tf_hi.txt"
            else:
                root_path = "results/web-webv1k_eval-pretrained-feat_save_FULL/stage1/WebVision"
                save_root = "dataset/web_meta/webvision_eval/image_graph"
                filelist_path = "filelist/val_webvision_1k_usable_tf_hi.txt"
        else:
            tfrecord_path = "dataset/webvision1k/tfrecord/tfrecord_webvision_train"
            root_path = "results/web-webv1k-pretrained-feat_save_FULL/stage1"
            # save_root = "dataset/webvision1k/image_graph"
            save_root = "dataset/web_meta/webvision/image_graph"
            if is_g500:
                filelist_path = "filelist/low_resolution/train_filelist_google_500_usable_tf.txt"
            else:
                filelist_path = "filelist/low_resolution/train_filelist_webvision_1k_usable_tf.txt"
        tfrecord2feats_all_id_path = os.path.join(root_path, "tfrecord2feats_all_id.json")

    feats_all_path = os.path.join(root_path, "feats_all.npy")

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
        knn_graph = global_build(feats_all, dist_def="cosine:", k=10,\
            save_filename=os.path.join(save_root, "nuswide_train_knn"), build_graph_method='gpu')
        save_vis_path = os.path.join(save_root, "image_visualize_knn_nn_nuswide")
    else:
        if is_g500:
            knn_graph = global_build(feats_all, dist_def="cosine:", k=5,\
                save_filename=os.path.join(save_root, "google500_train_knn"), build_graph_method='gpu')
            save_vis_path = os.path.join(save_root, "image_visualize_knn_nn_google500")
        else:
            knn_graph = global_build(feats_all, dist_def="cosine:", k=5,\
                save_filename=os.path.join(save_root, "webvision_train_knn"), build_graph_method='gpu')
            save_vis_path = os.path.join(save_root, "image_visualize_knn_nn_webvision")
    os.makedirs(save_vis_path, exist_ok=True)

    if is_visualize:
        for feat_id, knn_id in enumerate(knn_graph.knns[:20]):
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
            assert os.path.exists(meta_info_path), "{} exists".format(meta_info_path)
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
        np.save(os.path.join(root_path, "knn_smoothed_" + os.path.basename(meta_info_path)), text_feature)




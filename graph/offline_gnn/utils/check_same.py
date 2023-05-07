import json
import numpy as np
import os

# def load_mapping_tfrecord2embd_idx(mapping_tfrecord2embd_idx_path):
#     tf2embd_id = {}
#     with open(mapping_tfrecord2embd_idx_path, "r") as fr:
#         for line in fr.readlines():
#             info = line.strip().split("\t")
#             tfrecord_path = info[0]
#             embd_idx = int(info[1])
#             tf2embd_id[tfrecord_path] = embd_idx
#     return tf2embd_id

# tfrecord2vis_embd = "/dockerdata/yuleiqin/webvision/proposed_ckpt/web-webv1k-pretrained-feat_save_FULL/stage1/tfrecord2feats_all_id.json"
# tfrecord2txt_embd = "/youtu-reid/yuleiqin/data/webvision1k/meta_data_tf/mapping_tfrecord2embd_idx.txt"
# with open(tfrecord2vis_embd, "r") as fr:
#     mapping_tfrecord2vis_embd_idx = json.load(fr)
# mapping_tfrecord2text_embd_idx = load_mapping_tfrecord2embd_idx(tfrecord2txt_embd)
# print("visual", len(mapping_tfrecord2vis_embd_idx), "textual", len(mapping_tfrecord2text_embd_idx))
# for tfrecord_name in mapping_tfrecord2text_embd_idx.keys():
#     text_idx = int(mapping_tfrecord2text_embd_idx[tfrecord_name])
#     vis_idx = int(mapping_tfrecord2vis_embd_idx[tfrecord_name])
#     if text_idx != vis_idx:
#         print(tfrecord_name)

"""
## ReName/ReOrganize Features
"""
mapping_tfrecord2embd_idx_path = "/youtu-reid/yuleiqin/data/webvision1k/meta_data_tf/mapping_tfrecord2embd_idx.txt"
feat_path = "/youtu-reid/yuleiqin/data/webvision1k/meta_data_tf/knn_rerank_smoothed_meta_embd_xlnet.npy"

##text_feats_train
##text_rerank_feats_train

save_feat_path = "/youtu_pedestrian_detection/yuleiqin/web_meta/google500/text_features/xlnet/text_rerank_feats_train.npy"
save_mapping_tfrecord2embd_idx_path = "/youtu_pedestrian_detection/yuleiqin/web_meta/google500/tfrecord2feat_id.json"

feat_all = np.load(feat_path)
if mapping_tfrecord2embd_idx_path.endswith(".txt"):
    mapping_tfrecord2embd_idx = {}
    with open(mapping_tfrecord2embd_idx_path, "r") as fr:
        for line in fr.readlines():
            tfrecord_path, embd_idx = line.strip().split("\t")
            mapping_tfrecord2embd_idx[tfrecord_path] = int(embd_idx)
else:
    with open(mapping_tfrecord2embd_idx_path, "r") as fr:
        mapping_tfrecord2embd_idx = json.load(fr)

tfrecord_pathlist_path = "/youtu-reid/yuleiqin/code_utils/webFG-fewshot-research-v2/dataset/filelist/low_resolution/train_filelist_google_500_usable_tf.txt"

save_feat = []
save_mapping_tfrecord2embd_idx = {}

with open(tfrecord_pathlist_path, "r") as fr:
    for idx, line in enumerate(fr.readlines()):
        tfrecord_path = line.strip().split(" ")[0]
        feat_id = int(mapping_tfrecord2embd_idx[tfrecord_path])
        save_feat.append(feat_all[feat_id:feat_id+1])
        save_mapping_tfrecord2embd_idx[tfrecord_path] = idx

save_feat = np.concatenate(save_feat, axis=0)
np.save(save_feat_path, save_feat)
if not (os.path.exists(save_mapping_tfrecord2embd_idx_path)):
    with open(save_mapping_tfrecord2embd_idx_path, "w") as fw:
        json.dump(save_mapping_tfrecord2embd_idx, fw)


import numpy as np
import os
from glob import glob
from knn_utils import global_build, sgc_precompute
import json
import sys 
from nltk.corpus import wordnet as wn

"""
find similarity between class categories based on textual features (NUSwide)
"""

is_minilm = False
if is_minilm:
    model_name = "minilm"
    label_embeddings_path = "dataset/nus_wide_web/meta_data_tf/nuswide_wdnet_embd_minilm.npy"
else:
    model_name = "xlnet"
    label_embeddings_path = "dataset/nus_wide_web/meta_data_tf/nuswide_wdnet_embd_xlnet.npy"

feats_all = np.load(label_embeddings_path)
feats_all_norm = np.linalg.norm(feats_all, axis=1, keepdims=True)
feats_all /= feats_all_norm

save_root = "filelist"
root_path = os.path.dirname(label_embeddings_path)
dataset_name = "nuswide"
save_path = os.path.join(save_root, "label_sims_{}_{}.json".format(dataset_name, model_name))

knn_graph = global_build(feats_all, dist_def="cosine:", k=11,\
    save_filename=os.path.join(root_path, "knn_" + os.path.basename(label_embeddings_path)),\
        build_graph_method='gpu')

save_txt_path = os.path.join(root_path,\
    "sim_" + dataset_name + "_" + os.path.basename(label_embeddings_path).replace(".npy", ".txt"))
save_txt_name_path = os.path.join(root_path,\
    "sim_name_" + dataset_name + "_" + os.path.basename(label_embeddings_path).replace(".npy", ".txt"))

mapping_feat_id2wdnet_id_path = "dataset/nus_wide_web/meta_data_tf/mapping_wdnet2index.txt"
feat_id2wdnet_id = {}
with open(mapping_feat_id2wdnet_id_path, "r") as fr:
    for line in fr.readlines():
        feat_id, wdnet_id = line.strip().split("\t")
        feat_id2wdnet_id[int(feat_id)] = wdnet_id

label_sims = {}
with open(save_txt_path, "w") as fw:
    with open(save_txt_name_path, "w") as fw2:
        for feat_id, knn_id in enumerate(knn_graph.knns):
            ner, sim = knn_id
            wdnet_id = feat_id2wdnet_id[feat_id]

            pos_id, offset_id = wdnet_id[0], wdnet_id[1:]
            syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
            syn_name = "_".join(syn.lemma_names()[:2])
            
            info = [wdnet_id]
            info_name = [syn_name]
            info_id = []

            for ner_idx, sim_idx in zip(ner[1:], sim[1:]):
                wdnet_id_ner_idx = feat_id2wdnet_id[ner_idx]
                
                pos_id, offset_id = wdnet_id_ner_idx[0], wdnet_id_ner_idx[1:]
                syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
                syn_name = "_".join(syn.lemma_names()[:2])

                info.append(wdnet_id_ner_idx + "@" + "{:.4f}".format(sim_idx))
                info_name.append(syn_name + "@" + "{:.4f}".format(sim_idx))
                info_id.append((int(ner_idx), float("{:.4f}".format(sim_idx))))
            
            label_sims[int(feat_id)] = info_id
            fw.write("\t".join(info) + "\n")
            fw2.write("\t".join(info_name) + "\n")

with open(save_path, "w") as fw:
    json.dump(label_sims, fw)



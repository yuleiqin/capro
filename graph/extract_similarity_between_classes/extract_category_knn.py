import numpy as np
import os
from glob import glob
from knn_utils import global_build, sgc_precompute
import json
import sys 
from nltk.corpus import wordnet as wn

"""
find similarity between class categories based on textual features (WebVision/Google500/ImageNet)
"""

is_g500=True
label_embeddings_path = "dataset/webvision1k/meta_data_tf/imgnet_wdnet_embd_minilm.npy"
# label_embeddings_path = "dataset/webvision1k/meta_data_tf/imgnet_wdnet_embd_xlnet.npy"
feat_id2wdnet_id_path = "dataset/webvision1k/meta_data_tf/mapping_wdnet2index.txt"

root_path = os.path.dirname(label_embeddings_path)

feats_all = np.load(label_embeddings_path)
feats_all_norm = np.linalg.norm(feats_all, axis=1, keepdims=True)
feats_all /= feats_all_norm
feat_id2wdnet_id = {}

if is_g500:
    valid_index_path = "filelist/mapping_google_500.txt"
    valid_wdnet_ids = set()
    with open(valid_index_path, "r") as fr:
        for line in fr.readlines():
            _, wdnet_id = line.strip().split(" ")
            valid_wdnet_ids.add(wdnet_id)
    
    feats_all_g500 = []
    wdnet_ids_g500 = []
    with open(feat_id2wdnet_id_path, "r") as fr:
        for line in fr.readlines():
            index, wdnet_id = line.strip().split("\t")
            if wdnet_id in valid_wdnet_ids:
                wdnet_ids_g500.append(wdnet_id)
                feats_all_g500.append(feats_all[int(index)])
        
    assert(len(feats_all_g500) == 500 and len(wdnet_ids_g500) == 500)
    for index, wdnet_id in enumerate(wdnet_ids_g500):
        feat_id2wdnet_id[int(index)] = wdnet_id    
    feats_all = np.array(feats_all_g500)
    g500_suffix = "google500_"

else:
    with open(feat_id2wdnet_id_path, "r") as fr:
        for line in fr.readlines():
            index, wdnet_id = line.strip().split("\t")
            feat_id2wdnet_id[int(index)] = wdnet_id

    g500_suffix = ""

knn_graph = global_build(feats_all, dist_def="cosine:", k=6,\
    save_filename=os.path.join(root_path, "knn_" + g500_suffix + os.path.basename(label_embeddings_path)),\
        build_graph_method='gpu')

save_txt_path = os.path.join(root_path, "sim_" + g500_suffix + os.path.basename(label_embeddings_path).replace(".npy", ".txt"))
save_txt_name_path = os.path.join(root_path, "sim_name_" + g500_suffix + os.path.basename(label_embeddings_path).replace(".npy", ".txt"))

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

            for ner_idx, sim_idx in zip(ner[1:], sim[1:]):
                wdnet_id_ner_idx = feat_id2wdnet_id[ner_idx]

                pos_id, offset_id = wdnet_id_ner_idx[0], wdnet_id_ner_idx[1:]
                syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
                syn_name = "_".join(syn.lemma_names()[:2])

                info.append(wdnet_id_ner_idx + "@" + "{:.4f}".format(sim_idx))
                info_name.append(syn_name + "@" + "{:.4f}".format(sim_idx))
                
            fw.write("\t".join(info) + "\n")
            fw2.write("\t".join(info_name) + "\n")



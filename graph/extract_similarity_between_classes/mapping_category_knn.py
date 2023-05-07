import os
import json

"""
map category name (wordnet synset ID) to category ID (class id, int)
"""

is_g500 = False
is_minilm = False

if is_minilm:
    if is_g500:
        sim_path = "dataset/webvision1k/meta_data_tf/sim_google500_imgnet_wdnet_embd_minilm.txt"
    else:
        sim_path = "dataset/webvision1k/meta_data_tf/sim_imgnet_wdnet_embd_minilm.txt"
else:
    if is_g500:
        sim_path = "dataset/webvision1k/meta_data_tf/sim_google500_imgnet_wdnet_embd_xlnet.txt"
    else:
        sim_path = "dataset/webvision1k/meta_data_tf/sim_imgnet_wdnet_embd_xlnet.txt"

if is_g500:
    wdnet_id2label_path = "filelist/mapping_google_500.txt"
else:
    wdnet_id2label_path = "filelist/mapping_webvision_1k.txt"

wdnet_id2label = {}
with open(wdnet_id2label_path, "r") as fr:
    for line in fr.readlines():
        label_id, wdnet_id = line.strip().split(" ")
        wdnet_id2label[wdnet_id] = int(label_id)

save_root = "filelist"
if is_minilm:
    model_name = "minilm"
else:
    model_name = "xlnet"
if is_g500:
    dataset_name = "google_500"
else:
    dataset_name = "webvision_1k"

save_path = os.path.join(save_root, "label_sims_{}_{}.json".format(dataset_name, model_name))
label_sims = {}
with open(sim_path, "r") as fr:
    for line in fr.readlines():
        info = line.strip().split("\t")
        label_id = wdnet_id2label[info[0]]
        label_id_sims = []
        for sim_info in info[1:]:
            wdnet_id_i, sim_i = sim_info.strip().split("@")
            label_id_sims.append((wdnet_id2label[wdnet_id_i], float(sim_i)))
        label_sims[label_id] = label_id_sims
print(save_path, len(label_sims))
with open(save_path, "w") as fw:
    json.dump(label_sims, fw)

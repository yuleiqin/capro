import os
from tqdm import tqdm
from glob import glob
import json
import shutil
import random
import csv
import re
import json
import io
import html
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import string
import enchant
import struct
from copy import deepcopy
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet as wn
from english_words import english_words_set
dict_enUS = enchant.Dict("en_US")
dict_enUK = enchant.Dict("en_UK")
# nltk.download('words')
dict_enNLTK = set(nltk.corpus.words.words())
import torch
import numpy as np
from sentence_transformers.util import cos_sim
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoTokenizer, GPTNeoModel
from scipy.spatial.distance import cosine
import time
### 需要使用分词模型
import wordninja
from DataLoader.example_pb2 import Example
from graph.rerank_feature import re_ranking


"""继续开展的工作可以从利用meta信息入手考虑
"""
TOKENIZER = RegexpTokenizer(r'\w+')
BLACKLIST = get_stop_words('en')
# list(set(['wikipedia', 'google', 'flickr', 'figure', 'photo', 'image', 'picture', 'homepage',\
# 'url', 'youtube', 'bing', 'baidu', 'free', 'twitter', 'facebook', 'instagram',\
# 'photos', 'images', 'pictures', 'figures', 'tumblr', 'pinterest', 'discord',\
# 'meme', 'amazon', 'taobao', 'blog', 'quora', 'reddit', 'forum', 'ads', 'amp', 'de', 'of',\
# 'amazoncom', 'alibaba', 'com', 'alibabacom', 'ebay', 'ebaycom', 'amazones', 'amazonin',\
# 'amazoncouk', 'amazonco', 'amazonfr', 'amazonsg', 'aliexpress', "aed", "adead"]))
SUFFIX = set([".jpg", ".jpeg", ".png", ".gif"])
## 0代表FLICKR图像
## 1代表GOOGLE图像



def denoise_text(text):
    """this funtion is to denoise text and only keep useful info
    """
    text_denoised = deepcopy(text.lower())
    ## 移除图像后缀名
    for suffix_i in SUFFIX:
        text_denoised = text_denoised.replace(suffix_i, "")
    ## 替换转移符
    text_denoised = html.unescape(text_denoised)
    ## 删除多余的html标记
    text_denoised = re.sub(r'<.*?>', '', text_denoised)
    ## 删除网址元素
    text_denoised = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil|cn)[\S]*\s?',
        '', text_denoised)
    text_denoised = re.sub(r'http\S+', '', text_denoised)
    ## 删除换行以及缩进符
    text_denoised = text_denoised.replace("\t", "").replace("\n", "")
    ## 删除stop words以及特殊符号
    # text_denoised_items = text_denoised.strip().split(" ")
    text_denoised_items = TOKENIZER.tokenize(text_denoised)
    text_denoised = []
    for text_denoised_item in text_denoised_items:   
        ## 删除digit数字
        text_denoised_item = ''.join(i for i in text_denoised_item if not i.isdigit())
        ## 删除单个字母
        if len(text_denoised_item) <= 2:
            continue
        ## 删除非alphabetic单词
        if not (text_denoised_item.isalpha()):
            continue
        ## 删除黑名单里的词
        if text_denoised_item in BLACKLIST:
            continue
        ## 删除非英文单词
        if dict_enUS.check(text_denoised_item) or\
            dict_enUK.check(text_denoised_item) or\
            (text_denoised_item in dict_enNLTK) or\
            (text_denoised_item in english_words_set):
            text_denoised.append(text_denoised_item)
    text_denoised = " ".join(text_denoised)
    return text_denoised


def get_img_label():
    #### 获得所有图像样本的路径和标签
    img_pathlist_path = "dataset/nus_wide_web/ImageList/Imagelist.txt"
    img_root_path = "dataset/nus_wide_web/Flickr"

    save_root = "dataset/nus_wide_web/dataset_raw"
    os.makedirs(save_root, exist_ok=True)

    save_img_path = os.path.join(save_root, "img_pathlist.txt")
    save_img_mapping = os.path.join(save_root, "mapping_img2index.txt")
    save_img2idx_mapping = os.path.join(save_root, "mapping_img2idx.json")
    save_idx2img_mapping = os.path.join(save_root, "mapping_idx2img.json")

    save_label_mapping = os.path.join(save_root, "mapping_label2index.txt")
    save_label2idx_mapping = os.path.join(save_root, "mapping_label2idx.json")
    save_idx2label_mapping = os.path.join(save_root, "mapping_idx2label.json")

    save_class_by_img = os.path.join(save_root, "class_by_img.json")
    save_img_by_class = os.path.join(save_root, "img_by_class.json")
    save_not_visited = os.path.join(save_root, "not_visited.txt")

    label_root_path = "dataset/nus_wide_web/AllLabels"
    label_txt_pathlist = glob(os.path.join(label_root_path, "*.txt"))
    label_txt_pathlist = sorted(label_txt_pathlist, key=lambda x:os.path.basename(x))
    print(label_txt_pathlist)
    mapping_label2index = {}
    mapping_index2label = {}
    with open(save_label_mapping, "w") as fw:
        for label_index, label_path in enumerate(label_txt_pathlist):
            ### 记录下标签的下标index以及标签的名字
            label_name = os.path.basename(label_path).split(".txt")[0].split("_")[1]
            # print("label name {} with path {}".format(label_name, label_path))
            mapping_label2index[label_name] = str(label_index)
            mapping_index2label[label_index] = label_name
            fw.write("\t".join([str(label_index), str(label_name)]) + "\n")
    print("mapping label 2 index", mapping_label2index, len(mapping_label2index))
    print("mapping index 2 label", mapping_index2label, len(mapping_index2label))
    with open(save_label2idx_mapping, "w") as fw:
        json.dump(mapping_label2index, fw)
    with open(save_idx2label_mapping, "w") as fw:
        json.dump(mapping_index2label, fw)

    mapping_img2index = {}
    mapping_index2img = {}
    
    with open(save_img_path, "w") as fw1:
        with open(save_img_mapping, "w") as fw2:
            with open(img_pathlist_path, "r") as fr:
                for img_index, line in enumerate(fr.readlines()):
                    img_name = line.strip().replace("\\", "/")
                    img_path = os.path.join(img_root_path, img_name)
                    if not os.path.exists(img_path):
                        print(img_path)
                    assert(os.path.exists(img_path))
                    fw1.write(img_path + "\n")
                    fw2.write("\t".join([str(img_index), str(img_path)]) + "\n")
                    mapping_img2index[img_path] = str(img_index)
                    mapping_index2img[str(img_index)] = img_path
    with open(save_img2idx_mapping, "w") as fw:
        json.dump(mapping_img2index, fw)
    with open(save_idx2img_mapping, "w") as fw:
        json.dump(mapping_index2img, fw)

    class_by_imgs = {}
    imgs_by_class = {}
    visited_index = set()
    
    num_instances = 0
    for label_index, label_path in enumerate(label_txt_pathlist):
        if not (label_index in imgs_by_class):
            imgs_by_class[label_index] = []
        with open(label_path, "r") as fr:
            num_instances = 0
            for img_index, line in enumerate(fr.readlines()):
                ### 如果是1则为对应类别
                isTrue = (int(line.strip()) == 1)
                if isTrue:
                    imgs_by_class[label_index].append(img_index)
                    if not (img_index in class_by_imgs):
                        class_by_imgs[img_index] = []
                    class_by_imgs[img_index].append(label_index)
                    visited_index.add(img_index)
                num_instances += 1
    
    with open(save_not_visited, "w") as fw:
        for i in range(num_instances):
            if not (i in visited_index):
                fw.write(str(i) + "\n")

    with open(save_class_by_img, "w") as fw:
        json.dump(class_by_imgs, fw)

    with open(save_img_by_class, "w") as fw:
        json.dump(imgs_by_class, fw)
    return


def visualize():
    """extract classes samples for visualization
    """
    ## 抽几个类别visualize
    img_by_class_path = "dataset/nus_wide_web/dataset_raw/img_by_class.json"
    class_id2name_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2label.json"
    img_id2path_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2img.json"
    with open(img_by_class_path, "r") as fr:
        img_by_class = json.load(fr)
    with open(class_id2name_path, "r") as fr:
        class_id2name = json.load(fr)
    with open(img_id2path_path, "r") as fr:
        img_id2path = json.load(fr)

    save_root = "dataset/nus_wide_web"
    save_vis_root = os.path.join(save_root, "vis")
    os.makedirs(save_vis_root, exist_ok=True)
    sample_classes = random.sample(img_by_class.keys(), 10)
    for sample_class in sample_classes:
        class_name = class_id2name[sample_class]
        sample_class_path = os.path.join(save_vis_root, class_name)
        os.makedirs(sample_class_path, exist_ok=True)
        sample_img_idx_pathlist = img_by_class[sample_class][:10]
        for sample_img_idx in sample_img_idx_pathlist:
            img_path = img_id2path[str(sample_img_idx)]
            img_class_name = os.path.basename(os.path.dirname(img_path))
            shutil.copyfile(img_path, os.path.join(sample_class_path,\
                img_class_name + "_" + os.path.basename(img_path)))
    return


def filter_out_tags():
    """filter out irrelevant words
    """
    root_path = "dataset/nus_wide_web/dataset_web"
    os.makedirs(root_path, exist_ok=True)

    img_pathlist_path = "dataset/nus_wide_web/dataset_raw/img_pathlist.txt"
    class_id2name_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2label.json"
    with open(class_id2name_path, "r") as fr:
        class_id2name = json.load(fr)
    class_name2id_path = "dataset/nus_wide_web/dataset_raw/mapping_label2idx.json"
    with open(class_name2id_path, "r") as fr:
        class_name2id = json.load(fr)
    img_idx2path_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2img.json"
    with open(img_idx2path_path, "r") as fr:
        img_idx2path = json.load(fr)

    save_tags_by_img_idx_path = "dataset/nus_wide_web/dataset_web/tags_by_img.json"
    if not os.path.exists(save_tags_by_img_idx_path):
        save_mapping_suffix2idx_path = "dataset/nus_wide_web/dataset_web/mapping_suffix2index.json"
        if not (os.path.exists(save_mapping_suffix2idx_path)):
            mapping_img2idx_path = "dataset/nus_wide_web/dataset_raw/mapping_img2index.txt"
            ### 为了得到每张图像的suffix后缀名映射到对应的index
            mapping_suffix2idx = {}
            with open(mapping_img2idx_path, "r") as fr:
                for line in fr.readlines():
                    info = line.strip().split("\t")
                    img_index, img_path = info[0], info[1]
                    img_suffix = os.path.basename(img_path).split(".jpg")[0].split("_")[1]
                    mapping_suffix2idx[img_suffix] = img_index
            with open(save_mapping_suffix2idx_path, "w") as fw:
                json.dump(mapping_suffix2idx, fw)
        else:
            with open(save_mapping_suffix2idx_path, "r") as fr:
                mapping_suffix2idx = json.load(fr)

        all_tags_path = "dataset/nus_wide_web/All_Tags.txt"
        all_tags_by_img_idx = {}
        with open(all_tags_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split("      ")
                img_suffix = info[0]
                assert(img_suffix in mapping_suffix2idx)
                img_idx = mapping_suffix2idx[img_suffix]
                if len(info) < 2:
                    all_tags_by_img_idx[img_idx] = []
                else:
                    img_tags = info[1].strip().split(" ")
                    img_tags = [img_tag.strip() for img_tag in img_tags if (img_tag.strip() != "")]
                    all_tags_by_img_idx[img_idx] = img_tags
        with open(save_tags_by_img_idx_path, "w") as fw:
            json.dump(all_tags_by_img_idx, fw)
    else:
        with open(save_tags_by_img_idx_path, "r") as fr:
            all_tags_by_img_idx = json.load(fr)
    
    ### 严格的匹配机制
    web_class_by_img = {}
    save_web_label_json_path = os.path.join(root_path, "web_label_by_img.json")
    save_web_label_txt_path = os.path.join(root_path, "web_label_by_img.txt")
    count_valid_sample = 0
    with open(save_web_label_txt_path, "w") as fw:
        for img_index in all_tags_by_img_idx:
            ## 判断所有样本的tag是否符合要求
            img_path = img_idx2path[img_index]
            img_tags = all_tags_by_img_idx[img_index]
            img_dirname_raw = os.path.basename(os.path.dirname(img_path))
            img_labels_idx = []
            img_labels_names = []
            for img_tag in img_tags:
                if img_tag in class_name2id:
                    img_labels_idx.append(class_name2id[img_tag])
                    img_labels_names.append(img_tag)
            if len(img_labels_names) > 0:
                web_class_by_img[img_index] = img_labels_idx
                count_valid_sample += 1
                img_labels_names = ",".join(img_labels_names)
                fw.write("\t".join([img_index, img_dirname_raw, img_labels_names]) + "\n")

    print("number of {} valid images with valid web labels from tags".format(count_valid_sample))
    with open(save_web_label_json_path, "w") as fw:
        json.dump(web_class_by_img, fw)

    return


def load_official_tags():
    official_tag_path = "dataset/nus_wide_web/AllTags81.txt"
    class_id2name_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2label.json"
    with open(class_id2name_path, "r") as fr:
        class_id2name = json.load(fr)
    class_name2id_path = "dataset/nus_wide_web/dataset_raw/mapping_label2idx.json"
    with open(class_name2id_path, "r") as fr:
        class_name2id = json.load(fr)
    img_idx2path_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2img.json"
    with open(img_idx2path_path, "r") as fr:
        img_idx2path = json.load(fr)
    web_class_by_img = {}
    root_path = "dataset/nus_wide_web/dataset_web_official"
    save_web_label_json_path = os.path.join(root_path, "web_label_by_img.json")
    save_web_label_txt_path = os.path.join(root_path, "web_label_by_img.txt")
    count_valid_sample = 0
    with open(save_web_label_txt_path, "w") as fw:
        with open(official_tag_path, "r") as fr:
            for img_index, line in enumerate(fr.readlines()):
                img_index = str(img_index)
                img_path = img_idx2path[img_index]
                img_dirname_raw = os.path.basename(os.path.dirname(img_path))
                img_labels_idx = []
                img_labels_names = []
                info = line.strip().split(" ")
                assert(len(info) == 81)
                for idx, class_id in enumerate(info):
                    if int(class_id) == 1:
                        img_labels_idx.append(str(idx))
                        img_labels_names.append(class_id2name[str(idx)])
            
                if len(img_labels_names) > 0:
                    web_class_by_img[img_index] = img_labels_idx
                    count_valid_sample += 1
                    img_labels_names = ",".join(img_labels_names)
                    fw.write("\t".join([img_index, img_dirname_raw, img_labels_names]) + "\n")

    print("number of {} valid images with valid web labels from tags".format(count_valid_sample))
    with open(save_web_label_json_path, "w") as fw:
        json.dump(web_class_by_img, fw)
    return


def extract_imgs_class():
    web_label_json_path = "dataset/nus_wide_web/dataset_web_official/web_label_by_img.json"
    count_labels = {}
    with open(web_label_json_path, "r") as fr:
        web_label_json = json.load(fr)
    for img_index, img_labels in web_label_json.items():
        for img_label in img_labels:
            if not (img_label in count_labels):
                count_labels[img_label] = 0
            count_labels[img_label] += 1
    class_id2name_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2label.json"
    with open(class_id2name_path, "r") as fr:
        class_id2name = json.load(fr)
    root_path = "dataset/nus_wide_web/dataset_web_official"
    save_csv_path = os.path.join(root_path, "statistics_class.csv")
    with open(save_csv_path, "w") as fw:
        csv_writer = csv.writer(fw)
        for class_id in count_labels:
            line = [str(class_id), str(class_id2name[str(class_id)]), count_labels[class_id]]
            csv_writer.writerow(line)
    return


def extract_closest_synset_by_wdnet():
    label_txt_path = "dataset/nus_wide_web/dataset_raw/mapping_label2index.txt"
    class_label_names = []
    with open(label_txt_path, "r") as fr:
        for line in fr.readlines():
            info = line.strip().split("\t")
            class_label_names.append(info[1])

    root_path = "dataset/nus_wide_web/dataset_web_official"
    save_attr_txt_path = os.path.join(root_path, "mapping_concept_wdnet_syn.txt")
    with open(save_attr_txt_path, "w") as fw:
        for class_label_idx, class_label_name in enumerate(class_label_names):
            synsets = wn.synsets(class_label_name, pos=wn.NOUN)
            # synsets_str = [str(synset) for synset in synsets]
            # fw.write("\t".join([str(class_label_idx),
            #     str(class_label_name)] + synsets_str) + "\n")
            synset_closest = synsets[0]
            fw.write("\t".join([str(class_label_idx), str(class_label_name)] +\
                [str(synset_closest.pos()) + str(synset_closest.offset())]) + "\n")
    return
        

def extract_tags_valid():
    ### 从meta info中抽取有效信息
    all_tags_path = "dataset/nus_wide_web/All_Tags.txt"
    # save_path_denoised_txt = "dataset/nus_wide_web/dataset_web_official/web_meta_by_img.txt"
    # save_path_denoised_json = "dataset/nus_wide_web/dataset_web_official/web_meta_by_img.json"
    # mapping_suffix2index_path = "dataset/nus_wide_web/dataset_web_official/mapping_pathsuffix2index_valid.txt"

    save_path_denoised_txt = "dataset/nus_wide_web/dataset_raw/web_meta_by_img_gt.txt"
    save_path_denoised_json = "dataset/nus_wide_web/dataset_raw/web_meta_by_img_gt.json"
    mapping_suffix2index_path = "dataset/nus_wide_web/dataset_raw/mapping_pathsuffix2index.txt"
    mapping_index2suffix = {}
    with open(mapping_suffix2index_path, "r") as fr:
        for line in tqdm(fr.readlines()):
            suffix, index = line.strip().split("\t")
            mapping_index2suffix[index] = suffix

    valid_tags = {}
    with open(save_path_denoised_txt, "w") as fw:
        with open(all_tags_path, "r") as fr:
            for idx, line in enumerate(fr.readlines()):
                info = line.strip().split("      ")
                if len(info) == 2:
                    suffix, text = info
                else:
                    suffix = info[0]
                    text = ""

                index = str(idx)
                if index in mapping_index2suffix:
                    suffix_check = mapping_index2suffix[index]
                    assert(suffix_check == suffix)
                    if text != "":
                        text_tags_all = []
                        text_tags = text.split(" ")
                        for text_tag in text_tags:
                            ### 英文分词
                            text_tag_list = wordninja.split(text_tag)
                            ### 使用空格重新连接
                            text_tag = " ".join(text_tag_list)
                            denoised_text = denoise_text(text_tag)
                            if denoised_text != "":
                                text_tags_all.append(denoised_text)
                        denoised_text = " ".join(text_tags_all)
                    else:
                        denoised_text = text

                    valid_tags[index] = denoised_text
                    fw.write("\t".join([index, denoised_text]) + "\n")

    with open(save_path_denoised_json, "w") as fw:
        json.dump(valid_tags, fw)
    return 


def extract_embeddings(model_name, is_meta=True):
    """extract embeddings of web images
    -- model_name: the name of the model in use
    -- is_meta: whether use meta information
    """
    if model_name == "minilm":
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Load MiniLM successfully")
    elif model_name == "xlnet":
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetModel.from_pretrained('xlnet-base-cased')
        print("Load XLNet successfully")
    elif model_name == 'gpt':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
        if is_meta:
            model = model.to("cuda")
        print("Load GPT successfully")
    else:
        raise NotImplementedError("model name only supports minilm and xlnet now")

    root_dir = "dataset/nus_wide_web/meta_data_tf"
    os.makedirs(root_dir, exist_ok=True)

    if is_meta:
        web_label_by_img_path = "dataset/nus_wide_web/dataset_web_official/web_label_by_img.txt"
        save_txt_path = os.path.join(root_dir, "mapping_img_idx2embd_idx.txt")
        img_indexes = []
        with open(save_txt_path, "w") as fw:
            with open(web_label_by_img_path, "r") as fr:
                for idx, line in enumerate(fr.readlines()):
                    img_index = line.strip().split("\t")[0]
                    # img_index = line.strip().split(" ")[1]
                    img_indexes.append(img_index)
                    fw.write("\t".join([img_index, str(idx)]) + "\n")

        embeddings = []
        web_meta_by_img_path = "dataset/nus_wide_web/dataset_web_official/web_meta_by_img.json"
        with open(web_meta_by_img_path, "r") as fr:
            web_meta_by_img = json.load(fr)

        with torch.no_grad():
            for img_index in tqdm(img_indexes):
                text_raw = web_meta_by_img[img_index]
                if text_raw == "":
                    text_raw = " "
                if model_name == "minilm":
                    embeddings.append(model.encode(text_raw))
                elif model_name == "xlnet":
                    outputs = model(**tokenizer(text_raw, return_tensors="pt")).last_hidden_state.mean(1)
                    embeddings.append(outputs.detach().numpy()[0])
                elif model_name == "gpt":
                    text_raw = text_raw[:256]
                    outputs = model(**tokenizer(text_raw, return_tensors="pt").to("cuda")).last_hidden_state.cpu()
                    outputs = outputs.detach().mean(1)
                    embeddings.append(outputs.numpy()[0])

        embeddings = np.array(embeddings)
        print(embeddings.shape)
        embedding_path = os.path.join(root_dir, "meta_embd_{}.npy".format(model_name))
        np.save(embedding_path, embeddings)
    else:
        imgnet_meta_path = "dataset/nus_wide_web/dataset_web_official/mapping_concept_wdnet_syn.txt"
        wdnet_ids = []
        with open(imgnet_meta_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split("\t")
                wdnet_id = info[2]
                assert(wdnet_id.startswith(wdnet_id))
                wdnet_ids.append(wdnet_id)

        save_mapping_index_path = os.path.join(root_dir, "mapping_wdnet2index.txt")
        if not os.path.exists(save_mapping_index_path):
            ### 存储一下wdnet_id名以及对应的编号
            with open(save_mapping_index_path, "w") as fw:
                for idx, wdnet_id in enumerate(wdnet_ids):
                        fw.write("\t".join([str(idx), str(wdnet_id)]) + "\n")
        embeddings = []
        with torch.no_grad():
            for wdnet_id in tqdm(wdnet_ids):
                pos_id, offset_id = wdnet_id[0], wdnet_id[1:]
                syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
                ## 加入adjacent synsets以及domains
                syn_all = [syn] + syn.hypernyms() + syn.instance_hypernyms() +\
                syn.hyponyms() + syn.instance_hyponyms() +\
                syn.topic_domains() + syn.region_domains() + syn.usage_domains()
                text_raw = []
                for syn in syn_all:
                    ### 加入所有别名
                    for name in syn.lemma_names():
                        text_raw.append(name.lower().replace("_", " ").replace("-", " ").lower().strip())
                    ### 利用该类别的名称+定义作为文本
                    text_raw.append(syn.definition().lower().replace("_", " ").replace("-", " ").lower().strip())
                    ### 直接写现成的例句
                    for example in syn.examples():
                        text_raw.append(example.lower().replace("_", " ").replace("-", " ").lower().strip())

                text_raw = " ".join(text_raw)
                if model_name == "minilm":
                    embeddings.append(model.encode(text_raw))
                elif model_name == "xlnet": 
                    outputs = model(**tokenizer(text_raw, return_tensors="pt")).last_hidden_state.mean(1)
                    embeddings.append(outputs.detach().numpy()[0])
                elif model_name == "gpt":
                    text_raw = text_raw[:256]
                    outputs = model(**tokenizer(text_raw, return_tensors="pt")).last_hidden_state.detach().mean(1)
                    embeddings.append(outputs.numpy()[0])

        embeddings = np.array(embeddings)
        print(embeddings.shape)
        embedding_path = os.path.join(root_dir, "nuswide_wdnet_embd_{}.npy".format(model_name))
        np.save(embedding_path, embeddings)
    return


def split_train_val_set():
    img_pathlist_path = "dataset/nus_wide_web/dataset_raw/img_index_pathlist.txt"
    img_pathlist_train_path = "dataset/nus_wide_web/dataset_raw/img_index_pathlist_train.txt"
    img_pathlist_val_path = "dataset/nus_wide_web/dataset_raw/img_index_pathlist_val.txt"

    train_split_path = "dataset/nus_wide_web/ImageList/TrainImagelist.txt"
    val_split_path = "dataset/nus_wide_web/ImageList/TestImagelist.txt"
    train_split = set()
    val_split = set()

    with open(train_split_path, "r") as fr:
        for line in fr.readlines():
            train_split.add(line.strip().replace("\\", "/"))
    with open(val_split_path, "r") as fr:
        for line in fr.readlines():
            val_split.add(line.strip().replace("\\", "/"))
    
    with open(img_pathlist_train_path, "w") as fw_train:
        with open(img_pathlist_val_path, "w") as fw_val:
            with open(img_pathlist_path, "r") as fr:
                for line in fr.readlines():
                    img_path = line.strip().split(" ")[0]
                    img_basename = os.path.join(os.path.basename(os.path.dirname(img_path)),
                        os.path.basename(img_path))
                    if img_basename in train_split:
                        fw_train.write(line)
                    elif img_basename in val_split:
                        fw_val.write(line)
                    else:
                        print("{} does not belong to trainset or valset".format(img_path))
    return
    

def get_val_pathlist():
    ### get the image pathlist for validation set
    val_set_mapping = {}
    val_set_path = "dataset/nus_wide_web/dataset_raw/img_index_pathlist_val.txt"
    with open(val_set_path, "r") as fr:
        for line in fr.readlines():
            img_path, img_index = line.strip().split(" ")
            val_set_mapping[img_path] = img_index

    img_label_path = "dataset/nus_wide_web/dataset_raw/class_by_img.json"
    with open(img_label_path, "r") as fr:
        img_label = json.load(fr)

    img_index_tfrecord_path = "dataset/nus_wide_web/tfrecord/image_all_tfrecord.pathlist"
    save_path = "dataset/nus_wide_web/dataset_raw/val_nus_81_tf.txt"
    with open(save_path, "w") as fw:
        with open(img_index_tfrecord_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split(" ")
                img_path = info[0]
                if img_path in val_set_mapping:
                    tfrecord_path, tfrecord_offset = info[2], info[3]
                    tfrecord_info = tfrecord_path+"@"+tfrecord_offset
                    img_index_tf = str(val_set_mapping[img_path])
                    if not (img_index_tf in img_label):
                        ### 存在部分样本不包括任何类别
                        continue
                    img_label_info = img_label[img_index_tf]
                    img_label_info = ",".join([str(img_label_i) for img_label_i in img_label_info])
                    fw.write(" ".join([tfrecord_info, img_label_info]) + "\n")
    return


def get_train_pathlist():
    ### get the image pathlist for training set
    train_set_mapping = {}
    train_set_path = "dataset/nus_wide_web/dataset_raw/img_index_pathlist_train.txt"
    with open(train_set_path, "r") as fr:
        for line in fr.readlines():
            img_path, img_index = line.strip().split(" ")
            train_set_mapping[img_path] = img_index

    ### 使用网图标签
    img_label_path = "dataset/nus_wide_web/dataset_web_official/web_label_by_img.json"
    with open(img_label_path, "r") as fr:
        img_label = json.load(fr)

    ### 使用原始GT标签
    # img_all_labels_path = "dataset/nus_wide_web/dataset_raw/class_by_img.json"
    # img_label_path_web = "dataset/nus_wide_web/dataset_web/web_label_by_img.json"
    # with open(img_all_labels_path, "r") as fr:
    #     img_all_label = json.load(fr)
    # with open(img_label_path_web, "r") as fr:
    #     img_label_path_web = json.load(fr)
    # img_label = {}
    # for img_index_web in img_label_path_web:
    #     img_label[img_index_web] = img_all_label[img_index_web]
    
    mapping_label_id2name_path = "dataset/nus_wide_web/dataset_raw/mapping_idx2label.json"
    with open(mapping_label_id2name_path, "r") as fr:
        mapping_label_id2name = json.load(fr)

    img_index_tfrecord_path = "dataset/nus_wide_web/tfrecord/image_all_tfrecord.pathlist"

    save_path = "dataset/nus_wide_web/dataset_raw/train_nus_81_tf.txt"
    save_statistics_path = "dataset/nus_wide_web/dataset_web_official/statistics_class_train.csv"

    num_train_class = [0 for _ in range(81)]

    results = []
    with open(img_index_tfrecord_path, "r") as fr:
        for line in fr.readlines():
            info = line.strip().split(" ")
            img_path = info[0]
            if img_path in train_set_mapping:
                tfrecord_path, tfrecord_offset = info[2], info[3]
                tfrecord_info = tfrecord_path+"@"+tfrecord_offset
                img_index_tf = str(train_set_mapping[img_path])
                if not (img_index_tf in img_label):
                    ### 存在部分样本不包括任何类别
                    continue
                img_label_info = img_label[img_index_tf]
                img_label_info = [int(img_label_i) for img_label_i in img_label_info]
                for img_label_i in img_label_info:
                    num_train_class[img_label_i] += 1
                img_info = {}
                img_info["conf_score"] = 1.0
                img_info["label"] = img_label_info
                results.append([tfrecord_info, img_info])

    num_train_class = np.array(num_train_class) + 1.

    with open(save_statistics_path, "w") as fw:
        csv_writer = csv.writer(fw)
        for class_id, num_class in enumerate(num_train_class):
            csv_writer.writerow([class_id, mapping_label_id2name[str(class_id)], int(num_class)])
    
    num_class_median = np.median(num_train_class)
    ratio_train_class = num_class_median / num_train_class
    with open(save_path, "w") as fw:
        for result in results:
            tfrecord_info, img_info = result
            img_sample_weights = [ratio_train_class[class_i] for class_i in img_info["label"]]
            img_sample_weight_max = float(np.amax(img_sample_weights))
            img_info["sample_weight"] = img_sample_weight_max
            img_info_str = json.dumps(img_info)
            fw.write(" ".join([tfrecord_info, img_info_str]) + "\n")

    return


def get_train_pathlist_meta(save_root, model_name, is_smoothed=False, is_rerank=False, is_dist=False):
    ### get the pathlist tfrecord with meta info
    mapping_tfrecord2index_path = "dataset/nus_wide_web/meta_data_tf/tfrecord2imgidx.json"

    if not (os.path.exists(mapping_tfrecord2index_path)):
        mapping_index_tfrecord_path = "dataset/nus_wide_web/tfrecord/image_all_tfrecord.pathlist"
        mapping_tfrecord2index = {}
        with open(mapping_index_tfrecord_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split(" ")
                img_index = info[1]
                tfrecord_path = info[2]
                tfrecord_offset = info[3]
                tfrecord_info = tfrecord_path + "@" + tfrecord_offset
                mapping_tfrecord2index[tfrecord_info] = img_index
        with open(mapping_tfrecord2index_path, "w") as fw:
            json.dump(mapping_tfrecord2index, fw)
    else:
        with open(mapping_tfrecord2index_path, "r") as fr:
            mapping_tfrecord2index = json.load(fr)
    
    train_tfrecord_path = "dataset/nus_wide_web/dataset_raw/train_nus_81_tf.txt"
    with open(train_tfrecord_path, "r") as fr:
        train_tfrecord = fr.readlines()

    mapping_img_idx2embd_idx_path = "dataset/nus_wide_web/meta_data_tf/mapping_img_idx2embd_idx.txt"
    mapping_img_idx2embd_idx = {}
    with open(mapping_img_idx2embd_idx_path, "r") as fr:
        for line in fr.readlines():
            img_idx, embd_idx = line.strip().split("\t")
            embd_idx = int(embd_idx)
            mapping_img_idx2embd_idx[img_idx] = embd_idx

    if is_smoothed:
        if is_rerank:
            smoothed_str = "knn_rerank_smoothed_" 
        else:
            smoothed_str = "knn_smoothed_"
    else:
        smoothed_str = ""

    save_tfrecord_index_path = os.path.join(save_root, "train_nus_81_tf_{}meta_{}.txt".format(smoothed_str, model_name))
    
    if model_name == "minilm":
        nuswide_wdnet_embd_path = "dataset/nus_wide_web/meta_data_tf/nuswide_wdnet_embd_minilm.npy"
        if is_smoothed:
            if is_rerank:
                meta_embd_path = "dataset/nus_wide_web/meta_data_tf/knn_rerank_smoothed_meta_embd_minilm.npy"
            else:
                meta_embd_path = "dataset/nus_wide_web/meta_data_tf/knn_smoothed_meta_embd_minilm.npy"
        else:
            meta_embd_path = "dataset/nus_wide_web/meta_data_tf/meta_embd_minilm.npy"
        nuswide_wdnet_embd = np.load(nuswide_wdnet_embd_path)
        meta_embd = np.load(meta_embd_path)

    elif model_name == "xlnet":
        nuswide_wdnet_embd_path = "dataset/nus_wide_web/meta_data_tf/nuswide_wdnet_embd_xlnet.npy"
        if is_smoothed:
            if is_rerank:
                meta_embd_path = "dataset/nus_wide_web/meta_data_tf/knn_rerank_smoothed_meta_embd_xlnet.npy"
            else:
                meta_embd_path = "dataset/nus_wide_web/meta_data_tf/knn_smoothed_meta_embd_xlnet.npy"
        else:
            meta_embd_path = "dataset/nus_wide_web/meta_data_tf/meta_embd_xlnet.npy"
        nuswide_wdnet_embd = np.load(nuswide_wdnet_embd_path)
        meta_embd = np.load(meta_embd_path)

    elif model_name == "gpt":
        nuswide_wdnet_embd_path = "dataset/nus_wide_web/meta_data_tf/nuswide_wdnet_embd_gpt.npy"
        if is_smoothed:
            if is_rerank:
                meta_embd_path = "dataset/nus_wide_web/meta_data_tf/knn_rerank_smoothed_meta_embd_gpt.npy"
            else:
                meta_embd_path = "dataset/nus_wide_web/meta_data_tf/knn_smoothed_meta_embd_gpt.npy"
        else:
            meta_embd_path = "dataset/nus_wide_web/meta_data_tf/meta_embd_gpt.npy"
        nuswide_wdnet_embd = np.load(nuswide_wdnet_embd_path)
        meta_embd = np.load(meta_embd_path)
    else:
        raise ValueError("current model name only supports minilm or xlnet")

    nuswide_wdnet_embd_norm = np.linalg.norm(nuswide_wdnet_embd, axis=1, keepdims=True)
    meta_embd_norm = np.linalg.norm(meta_embd, axis=1, keepdims=True)
    nuswide_wdnet_embd = nuswide_wdnet_embd / nuswide_wdnet_embd_norm
    meta_embd = meta_embd / meta_embd_norm
    print("meta embedding shape", meta_embd.shape)
    all_json_info = []
    tfrecord_info_by_label_id = {}

    tfrecordfeat_id_json_path = "dataset/nus_wide_web/meta_data_tf/tfrecord2feat_id.json"
    with open(tfrecordfeat_id_json_path, "r") as fr:
        mapping_tfrecord2embd_idx = json.load(fr)

    for line in tqdm(train_tfrecord):
        tfrecord_info = line.strip().split(" ")[0]
        json_str = line.strip().replace(tfrecord_info+" ", "")
        img_json = json.loads(json_str)
        img_labels = img_json["label"]
        # img_index = mapping_tfrecord2index[tfrecord_info]
        # embd_index = mapping_img_idx2embd_idx[img_index]
        embd_index = mapping_tfrecord2embd_idx[tfrecord_info]

        embd_tf = meta_embd[embd_index]
        ### meta_label ==> top5
        ### meta_sim ==> top5
        sim_cos = np.matmul(embd_tf, nuswide_wdnet_embd.T)
        assert(len(sim_cos) == 81)

        img_meta_labels = [img_label for img_label in img_labels]
        img_meta_sims = [float(sim_cos[img_label]) for img_label in img_labels]

        for img_label in img_meta_labels:
            if not (img_label in tfrecord_info_by_label_id):
                tfrecord_info_by_label_id[img_label] = []
            tfrecord_info_by_label_id[img_label].append(tfrecord_info)
        
        img_json["meta_label"] = img_meta_labels
        img_json["meta_sim"] = img_meta_sims

        ### sort the cosine similarity from maximum to minimum
        sim_cos_argmax = np.argsort(sim_cos).tolist()[::-1]
        
        img_top10_labels = []
        img_top10_sims = []

        for sim_cos_top_id in sim_cos_argmax[:10]:
            img_top10_labels.append(sim_cos_top_id)
            img_top10_sims.append(float(sim_cos[sim_cos_top_id]))
    
        img_json["meta_label_top10"] = img_top10_labels
        img_json["meta_sim_top10"] = img_top10_sims
        all_json_info.append([tfrecord_info, img_json])

    if is_dist:
        ## 计算每个样本经过各类别rerank后的距离
        dist_by_tfrecord_path = {}
        
        for label_id in tqdm(tfrecord_info_by_label_id.keys()):
            tfrecord_infolist = tfrecord_info_by_label_id[label_id]
            embd_wdnet = nuswide_wdnet_embd[label_id:label_id+1]  # probFeat

            embd_tf_list = []
            for tfrecord_info in tfrecord_infolist:
                # img_index = mapping_tfrecord2index[tfrecord_info]
                # embd_index = mapping_img_idx2embd_idx[img_index]
                embd_index = mapping_tfrecord2embd_idx[tfrecord_info]
                embd_tf = meta_embd[embd_index:embd_index+1]
                embd_tf_list.append(embd_tf)
            embd_tf_list = np.concatenate(embd_tf_list, axis=0)  # galleryFeat
            
            dist_rerank = re_ranking(embd_wdnet, embd_tf_list,\
                MemorySave=(embd_tf_list.shape[0] > 10000))[0]  # only one prob feature
            
            for tfrecord_info, dist in zip(tfrecord_infolist, dist_rerank):
                if not (tfrecord_info in dist_by_tfrecord_path):
                    dist_by_tfrecord_path[tfrecord_info] = {}
                dist_by_tfrecord_path[tfrecord_info][label_id] = dist
        
        with open(save_tfrecord_index_path, "w") as fw:
            for info in all_json_info:
                tfrecord_info, img_json = info
                meta_dist = []
                for label_id in img_json["meta_label"]:
                    dist_id = dist_by_tfrecord_path[tfrecord_info][label_id]
                    meta_dist.append(float(dist_id))
                img_json['meta_dist'] = meta_dist
                json_str_new = json.dumps(img_json)
                fw.write(" ".join([tfrecord_info, json_str_new]) + "\n")                

    else:
        ## 直接计算的是每个样本与各类别描述值的点积cosine相似度
        with open(save_tfrecord_index_path, "w") as fw:
            for info in all_json_info:
                tfrecord_info, img_json = info
                meta_dist= []
                for label_id, sim_id in zip(img_json["meta_label"], img_json["meta_sim"]):
                    dist_id = 1.-sim_id
                    assert(dist_id>=0 and dist_id<=1)
                    meta_dist.append(dist_id)
                img_json['meta_dist'] = meta_dist
                json_str_new = json.dumps(img_json)
                fw.write(" ".join([tfrecord_info, json_str_new]) + "\n")
    return


def get_tfrecord_image(record_file, offset):
    """read images from tfrecord"""

    def parser(feature_list):
        """get the image file and perform transformation
        feature_list: the dictionary to load features (images)
        return: PIL Image object
        """
        for key, feature in feature_list: 
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = Image.open(io.BytesIO(image_raw))
                image = image.convert('RGB')
                return image
        return

    with open(record_file, 'rb') as ifs:
        ifs.seek(offset)
        byte_len_crc = ifs.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        pb_data = ifs.read(proto_len)
        if len(pb_data) < proto_len:
            print("read pb_data err, proto_len:%s pb_data len:%s" % (proto_len, len(pb_data)))
            return
    example = Example()
    example.ParseFromString(pb_data)
    # keep key value in order
    feature = sorted(example.features.feature.items())
    image = parser(feature)
    return image


def extract_top_k_by_cosine_similarity(model_name, save_root, tfrecord_root,\
    is_smoothed=False, is_rerank=False, is_dist=False):
    """extract top k samples by the cosine similarity between meta info and class_text
    model_name: the name of model to encode sentences
    save_root: the directory path to save images
    tfrecord_root: the directory path to load tfrecord images
    """
    if is_smoothed:
        if is_rerank:
            smoothed_str = "knn_rerank_smoothed_" 
        else:
            smoothed_str = "knn_smoothed_"
    else:
        smoothed_str = ""

    timestr = time.strftime("%m%d_%H%M%S")
    save_root = os.path.join(save_root, model_name + "_vis_{}{}".format(smoothed_str, timestr))
    os.makedirs(save_root, exist_ok=True)

    tfrecord_index_path = "dataset/nus_wide_web/dataset_raw/train_nus_81_tf_{}meta_{}.txt".format(smoothed_str, model_name)
    print("tfrecord_index_path:", tfrecord_index_path)
    assert(os.path.exists(tfrecord_index_path)), "the path to tfrecord index must exist"

    class_id2wdnet_id = {}
    wdnet_ids = []
    class_id2wdnet_id_path = "dataset/nus_wide_web/meta_data_tf/mapping_wdnet2index.txt"
    with open(class_id2wdnet_id_path, "r") as fr:
        for line in fr.readlines():
            info = line.strip().split("\t")
            class_idx = int(info[0])
            wdnet_id = info[1]
            class_id2wdnet_id[class_idx] = wdnet_id
            wdnet_ids.append(wdnet_id)

    embd_sim = {}
    with open(tfrecord_index_path, "r") as fr:
        for line in tqdm(fr.readlines()):
            info = line.strip()
            tfrecord_path = info.split(" ")[0]
            json_str = info.replace(tfrecord_path+" ", "")
            json_info = json.loads(json_str)
            meta_label = json_info['meta_label']
            if is_dist:
                meta_sim = json_info['meta_dist']
            else:
                meta_sim = json_info['meta_sim']
            for meta_label_i, meta_sim_i in zip(meta_label, meta_sim):
                meta_wdnet_i = class_id2wdnet_id[int(meta_label_i)]
                meta_sim_i = float(meta_sim_i)
                if not (meta_wdnet_i in embd_sim):
                    embd_sim[meta_wdnet_i] = []
                embd_sim[meta_wdnet_i].append([tfrecord_path, meta_sim_i])

    for wdnet_id in tqdm(wdnet_ids[:10]):
        pos_id, offset_id = wdnet_id[0], wdnet_id[1:]
        syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
        save_dir_name = wdnet_id + "_" + "_".join(syn.lemma_names()[:2]).replace("-", "_").replace(" ", "_")
        save_dir_path = os.path.join(save_root, save_dir_name)
        os.makedirs(save_dir_path, exist_ok=True)
        if is_dist:
            tfrecord_pathlist = sorted(embd_sim[wdnet_id], key=lambda x:x[1], reverse=False)
        else:
            tfrecord_pathlist = sorted(embd_sim[wdnet_id], key=lambda x:x[1], reverse=True)
        tfrecord_pathlist_top_sim = tfrecord_pathlist[:50]
        tfrecord_pathlist_btm_sim = tfrecord_pathlist[-50:]
        for tfrecord_item in tfrecord_pathlist_top_sim:
            tfrecord_path = tfrecord_item[0]
            tfrecord_name, tfrecord_offset = tfrecord_path.strip().split("@")
            tfrecord_name_path = os.path.join(tfrecord_root, tfrecord_name)
            assert(os.path.exists(tfrecord_name_path))
            img = get_tfrecord_image(tfrecord_name_path, int(tfrecord_offset))
            sim_cos = tfrecord_item[1]
            save_img_path = os.path.join(save_dir_path, "top_" + str(tfrecord_path) + "_%.2f.jpg"%(sim_cos))
            img.save(save_img_path)
    
        for tfrecord_item in tfrecord_pathlist_btm_sim:
            tfrecord_path = tfrecord_item[0]
            tfrecord_name, tfrecord_offset = tfrecord_path.strip().split("@")
            tfrecord_name_path = os.path.join(tfrecord_root, tfrecord_name)
            assert(os.path.exists(tfrecord_name_path))
            img = get_tfrecord_image(tfrecord_name_path, int(tfrecord_offset))
            sim_cos = tfrecord_item[1]
            save_img_path = os.path.join(save_dir_path, "btm_" + str(tfrecord_path) + "_%.2f.jpg"%(sim_cos))
            img.save(save_img_path)
    return


def generate_mapping_tfrecord2embd_idx():
    mapping_tfrecord2index_path = "dataset/nus_wide_web/meta_data_tf/tfrecord2imgidx.json"
    with open(mapping_tfrecord2index_path, "r") as fr:
        mapping_tfrecord2index = json.load(fr)
    mapping_img_idx2embd_idx_path = "dataset/nus_wide_web/meta_data_tf/mapping_img_idx2embd_idx.txt"
    mapping_img_idx2embd_idx = {}
    with open(mapping_img_idx2embd_idx_path, "r") as fr:
        for line in fr.readlines():
            img_idx, embd_idx = line.strip().split("\t")
            embd_idx = int(embd_idx)
            mapping_img_idx2embd_idx[img_idx] = embd_idx
    save_mapping_tfrecord2embd_idx_path = "dataset/nus_wide_web/meta_data_tf/mapping_tfrecord2embd_idx.txt"
    with open(save_mapping_tfrecord2embd_idx_path, "w") as fw:
        for tfrecord, img_idx in mapping_tfrecord2index.items():
            if not (img_idx in mapping_img_idx2embd_idx):
                print("image index {} does not exist".format(img_idx))
                continue
            fw.write("\t".join([tfrecord, str(int(mapping_img_idx2embd_idx[img_idx]))]) + "\n")
    return


if __name__ == "__main__":
    get_img_label()
    filter_out_tags()
    load_official_tags()
    extract_imgs_class()
    extract_closest_synset_by_wdnet()

    print("processing meta info")
    extract_embeddings(model_name="minilm", is_meta=False)
    extract_embeddings(model_name="xlnet", is_meta=False)
    extract_embeddings(model_name="minilm", is_meta=True)
    extract_embeddings(model_name="xlnet", is_meta=True)
    extract_embeddings(model_name="gpt", is_meta=False)
    extract_embeddings(model_name="gpt", is_meta=True)

    split_train_val_set()
    get_val_pathlist()
    get_train_pathlist()

    # save_root = "dataset/nus_wide_web/dataset_raw"
    # get_train_pathlist_meta(save_root, "minilm")
    # get_train_pathlist_meta(save_root, "xlnet")
    # get_train_pathlist_meta(save_root, "gpt")

    save_root = "dataset/nus_wide_web/dataset_raw_dist"
    os.makedirs(save_root, exist_ok=True)
    get_train_pathlist_meta(save_root, "minilm", is_smoothed=True, is_rerank=False, is_dist=True)
    get_train_pathlist_meta(save_root, "xlnet", is_smoothed=True, is_rerank=False, is_dist=True)
    get_train_pathlist_meta(save_root, "gpt", is_smoothed=True, is_rerank=False, is_dist=False)

    get_train_pathlist_meta(save_root, "minilm", is_smoothed=True, is_rerank=True, is_dist=True)
    get_train_pathlist_meta(save_root, "xlnet", is_smoothed=True, is_rerank=True, is_dist=True)
    get_train_pathlist_meta(save_root, "gpt", is_smoothed=True, is_rerank=True, is_dist=True)

    tfrecord_root = "dataset/nus_wide_web/tfrecord"
    # save_root = "dataset/nus_wide_web/results/cosine_similarity"
    # os.makedirs(save_root, exist_ok=True)
    # extract_top_k_by_cosine_similarity("minilm", save_root, tfrecord_root)
    # extract_top_k_by_cosine_similarity("minilm", save_root, tfrecord_root, is_smoothed=True)
    
    # extract_top_k_by_cosine_similarity("xlnet", save_root, tfrecord_root)
    # extract_top_k_by_cosine_similarity("xlnet", save_root, tfrecord_root, is_smoothed=True)

    save_root = "dataset/nus_wide_web/results/rerank_dist"
    os.makedirs(save_root, exist_ok=True)
    extract_top_k_by_cosine_similarity("minilm", save_root, tfrecord_root, is_smoothed=True, is_rerank=False, is_dist=True)
    extract_top_k_by_cosine_similarity("xlnet", save_root, tfrecord_root, is_smoothed=True, is_rerank=False, is_dist=True)
    extract_top_k_by_cosine_similarity("minilm", save_root, tfrecord_root, is_smoothed=True, is_rerank=True, is_dist=True)
    extract_top_k_by_cosine_similarity("xlnet", save_root, tfrecord_root, is_smoothed=True, is_rerank=True, is_dist=True)
    extract_top_k_by_cosine_similarity("gpt", save_root, tfrecord_root, is_smoothed=True, is_rerank=False, is_dist=False)
    extract_top_k_by_cosine_similarity("gpt", save_root, tfrecord_root, is_smoothed=True, is_rerank=True, is_dist=True)

    # generate_mapping_tfrecord2embd_idx()

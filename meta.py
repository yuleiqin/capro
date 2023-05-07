import os
import json
import re
from tqdm import tqdm
from copy import deepcopy
import html
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import string
import nltk
import enchant
import struct
from PIL import Image, ImageFile
import io
# from spellchecker import SpellChecker
# dict_spell = SpellChecker()
from english_words import english_words_set
dict_enUS = enchant.Dict("en_US")
dict_enUK = enchant.Dict("en_UK")
#### 使用现成的工具提取特征
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
import torch
import numpy as np
import random
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoTokenizer, GPTNeoModel
from scipy.spatial.distance import cosine
from DataLoader.example_pb2 import Example
import time
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
    nltk.download('words')
    dict_enNLTK = set(nltk.corpus.words.words())
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


def preprocess():
    """首先抽取meta info进行对齐tfrecord代码编号与原始Json文本代号
    """
    img_index_path = "dataset/webvision1k/tfrecord_webvision_train/image_all_tfrecord_google.pathlist"
    # img_index_path = "dataset/webvision1k/tfrecord_webvision_train/image_all_tfrecord_flickr.pathlist"
    root_path = "dataset/webvision1k/"
    root_meta = "dataset/webvision1k/meta_data/"
    root_save = "dataset/webvision1k/meta_data_tf"
    #### google/q1360/X0p-Y6iUfgD2vM.jpg
    #### 00052-of-00064.tfrecord@101027009
    meta_json_tf = {}
    save_json_path = os.path.join(root_save, "meta_data.json")
    with open(img_index_path, "r") as fr:
        for line in tqdm(fr.readlines()):
            info = line.strip().split(" ")
            img_path = info[0]
            wdnet_id = str(info[1])
            tfrecord_name = info[2]
            tfrecord_offset = info[3]
            tfrecord_path = str(tfrecord_name) + "@" + str(tfrecord_offset)
            img_name_base = img_path.replace(root_path, "")
            assert(not img_name_base.endswith("/"))
            # print("image basename", img_name_base)
            img_name = os.path.basename(img_name_base).split(".jpg")[0].split(".png")[0]
            dir_name = os.path.dirname(img_name_base)
            # print("image name {} dirname {}".format(img_name, dir_name))
            meta_json_path = os.path.join(root_meta, dir_name + ".json")
            if not os.path.exists(meta_json_path):
                print("meta json path", meta_json_path)
            assert(os.path.exists(meta_json_path))
            #### check load 并不是所有属性值都存在
            with open(meta_json_path, "r") as fr:
                meta_json = json.load(fr)
                item_meta = None
                for item_info in meta_json:
                    if item_info['id'] == img_name:
                        item_meta = item_info
                        break
                    else:
                        continue
                if item_meta is None:
                    raise ValueError("Not valid meta json info for {}".format(meta_json_path))
                meta_json_info = {}
                title = []
                if 'title' in item_meta:
                    title.append(denoise_text(item_meta['title']))
                description = []
                if 'description' in item_meta:
                    description.append(denoise_text(item_meta['description']))
                if 'tag' in item_meta:
                    description.append(denoise_text(item_meta['tags']))
                text = title + description
                text = [text_item for text_item in text if len(text_item) > 0]
                meta_json_info['text'] = " ".join(text)
                meta_json_info['rank'] = int(item_meta['rank'])
                meta_json_info['wdnet'] = wdnet_id
                if "flickr" in dir_name:
                    meta_json_info['source'] = 0
                elif "google" in dir_name:
                    meta_json_info['source'] = 1
                else:
                    raise ValueError("Invalid image path")
            meta_json_tf[tfrecord_path] = meta_json_info
        with open(save_json_path, "w") as fw:
            json.dump(meta_json_tf, fw)
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
    elif model_name == "gpt":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = model.to("cuda")
        print("Load GPT successfully")
    else:
        raise NotImplementedError("model name only supports minilm and xlnet and gpt now")

    root_dir = "dataset/webvision1k/meta_data_tf"
    if is_meta:
        # json_path = "dataset/webvision1k/meta_data_tf/meta_data.json"
        json_path = os.path.join(root_dir, "meta_data.json")
        with open(json_path, "r") as fr:
            meta_data = json.load(fr)
        save_mapping_index_path = os.path.join(root_dir, "mapping_tfrecord2index.txt")
        tfrecord_names = [key for key in meta_data.keys()]
        if not os.path.exists(save_mapping_index_path):
            ### 存储一下tfrecord名以及对应的编号
            with open(save_mapping_index_path, "w") as fw:
                for idx, tfrecord_name in enumerate(tfrecord_names):
                        fw.write("\t".join([str(idx), str(tfrecord_name)]) + "\n")
        embeddings = []
        with torch.no_grad():
            idx = 0
            for tfrecord_name in tqdm(tfrecord_names):
                text_raw = meta_data[tfrecord_name]['text']
                if text_raw == "":
                    text_raw = " "
                if model_name == "minilm":
                    embeddings.append(model.encode(text_raw))
                elif model_name == "xlnet":
                    outputs = model(**tokenizer(text_raw, return_tensors="pt")).last_hidden_state.mean(1)
                    embeddings.append(outputs.detach().numpy()[0])
                elif model_name == "gpt":
                    text_raw = text_raw[:256]
                    outputs = model(**tokenizer(text_raw, return_tensors="pt").to("cuda")).last_hidden_state.mean(1)
                    embedding = outputs.detach().cpu().numpy()[0]
                    embeddings.append(embedding)
                idx += 1
        embeddings = np.array(embeddings)
        print(embeddings.shape)
        embedding_path = os.path.join(root_dir, "meta_embd_{}.npy".format(model_name))
        np.save(embedding_path, embeddings)
    else:
        imgnet_meta_path = "filelist/imagenet_class_1k_full_index.json"
        with open(imgnet_meta_path, "r") as fr:
            imgnet_meta = json.load(fr)
        imgnet_id, imgnet_name, imgnet_wdnet = imgnet_meta
        wdnet_ids = [wdnet_id for wdnet_id in imgnet_wdnet.keys()]
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
                    ### 
                text_raw = " ".join(text_raw)
                if model_name == "minilm":
                    embeddings.append(model.encode(text_raw))
                elif model_name == "xlnet": 
                    outputs = model(**tokenizer(text_raw, return_tensors="pt")).last_hidden_state.mean(1)
                    embeddings.append(outputs.detach().numpy()[0])
                elif model_name == "gpt":
                    text_raw = text_raw[:256]
                    outputs = model(**tokenizer(text_raw, return_tensors="pt").to("cuda")).last_hidden_state.mean(1)
                    embeddings.append(outputs.detach().cpu().numpy()[0])

        embeddings = np.array(embeddings)
        print(embeddings.shape)
        embedding_path = os.path.join(root_dir, "imgnet_wdnet_embd_{}.npy".format(model_name))
        np.save(embedding_path, embeddings)
    return


def calculate_cosine_similarity(model_name, isTop5=False, isG500=False):
    """
    this function calculates the cosine similarity between each image's meta data and the class_name
    """
    root_dir = "dataset/webvision1k/meta_data_tf"
    tfrecord2wdnet_path = os.path.join(root_dir, "mapping_tfrecord2wdnet.txt")
    img_root_path = "dataset/webvision1k/tfrecord_webvision_train"
    if not os.path.exists(tfrecord2wdnet_path):
        print("Preprocess Mapping from TFRecord to WordNetID")
        img_index_path = os.path.join(img_root_path, "image_all_tfrecord_google.pathlist")
        # img_index_path = os.path.join(img_root_path, "image_all_tfrecord_flickr.pathlist")
        tfrecord_by_wdnet = {}
        with open(img_index_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split(" ")
                img_path = info[0]
                wdnet_id = str(info[1])
                tfrecord_name = info[2]
                tfrecord_offset = info[3]
                tfrecord_path = str(tfrecord_name) + "@" + str(tfrecord_offset)
                if not (wdnet_id in tfrecord_by_wdnet):
                    tfrecord_by_wdnet[wdnet_id] = []
                tfrecord_by_wdnet[wdnet_id].append(tfrecord_path)
        
        with open(tfrecord2wdnet_path, "w") as fw:
            for wdnet_id, tfrecord_pathlist in tfrecord_by_wdnet.items():
                for tfrecord_path in tfrecord_pathlist:
                    fw.write("\t".join([tfrecord_path, wdnet_id]) + "\n")
    else:
        print("Load Mapping from TFRecord to WordNetID")
        tfrecord_by_wdnet = {}
        with open(tfrecord2wdnet_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split("\t")
                tfrecord_path = info[0]
                wdnet_id = info[1]
                if not (wdnet_id in tfrecord_by_wdnet):
                    tfrecord_by_wdnet[wdnet_id] = []
                tfrecord_by_wdnet[wdnet_id].append(tfrecord_path)
    print("Load Mapping from Embd. to Index")
    web_embd_tf2id_path = os.path.join(root_dir, "mapping_tfrecord2index.txt")
    web_embd_tf2id = {}
    with open(web_embd_tf2id_path, "r") as fr:
        for line in fr.readlines():
            info = line.strip().split("\t")
            tfrecord_idx = info[0]
            tfrecord_path = info[1]
            web_embd_tf2id[tfrecord_path] = int(tfrecord_idx)

    imgnet_embd_wdnet2id_path = os.path.join(root_dir, "mapping_wdnet2index.txt")
    imgnet_embd_wdnet2id = {}
    imgnet_embd_id2wdnet = {}
    with open(imgnet_embd_wdnet2id_path, "r") as fr:
        for line in fr.readlines():
            wdnet_idx, wdnet_id = line.strip().split("\t")
            imgnet_embd_wdnet2id[wdnet_id] = int(wdnet_idx)
            imgnet_embd_id2wdnet[int(wdnet_idx)] = wdnet_id
    
    # web_embd_path = os.path.join(root_dir, "meta_embd_{}.npy".format(model_name))
    web_embd_path = os.path.join(root_dir, "knn_smoothed_meta_embd_{}.npy".format(model_name))
    imgnet_embd_path = os.path.join(root_dir, "imgnet_wdnet_embd_{}.npy".format(model_name))
    web_embd = np.load(web_embd_path)
    imgnet_embd = np.load(imgnet_embd_path)
    if not isTop5:
        save_embd_sim_txt_path = os.path.join(root_dir, "meta_imgnet_sim_{}.txt".format(model_name))
        save_embd_sim_json_path = os.path.join(root_dir, "meta_imgnet_sim_{}.json".format(model_name))
        embd_sim = {}
        with open(save_embd_sim_txt_path, "w") as fw:
            for wdnet_id in tqdm(list(tfrecord_by_wdnet.keys())):
                if not (wdnet_id in embd_sim):
                    embd_sim[wdnet_id] = []
                pos_id, offset_id = wdnet_id[0], wdnet_id[1:]
                syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
                print("wdnet id = {} names = {}".format(
                    wdnet_id,
                    syn.lemma_names(),
                    ))
                embd_wdnet = imgnet_embd[imgnet_embd_wdnet2id[wdnet_id]]
                tfrecord_pathlist = tfrecord_by_wdnet[wdnet_id]
                for tfrecord_path in tfrecord_pathlist:
                    embd_tf = web_embd[web_embd_tf2id[tfrecord_path]]
                    sim_cos = float(1. - cosine(embd_tf, embd_wdnet))
                    fw.write("\t".join([tfrecord_path, wdnet_id, "%.4f"%(sim_cos)]) + "\n")
                    embd_sim[wdnet_id].append([tfrecord_path, str(sim_cos)])
        with open(save_embd_sim_json_path, "w") as fw:
            json.dump(embd_sim, fw)
    else:
        #### 计算top 4相似度概率 & 对应原始网图的相似度概率
        web_embd_norm = np.linalg.norm(web_embd, axis=1, keepdims=True)
        imgnet_embd_norm = np.linalg.norm(imgnet_embd, axis=1, keepdims=True)
        web_embd = web_embd / web_embd_norm
        imgnet_embd = imgnet_embd / imgnet_embd_norm
        imgnet_embd = imgnet_embd.T
        if isG500:
            save_embd_sim_txt_path = os.path.join(root_dir, "meta_imgnet_sim_{}_g500_top5.txt".format(model_name))
            save_embd_sim_json_path = os.path.join(root_dir, "meta_imgnet_sim_{}_g500_top5.json".format(model_name))
            g500_valid_wdnet_ids = set()
            g500_valid_path = "filelist/mapping_google_500.txt"
            with open(g500_valid_path, "r") as fr:
                for line in fr.readlines():
                    wdnet_id = line.strip().split(" ")[1]
                    g500_valid_wdnet_ids.add(wdnet_id)
            print(g500_valid_wdnet_ids)
            print("number of valid wordnet ids = {}".format(len(g500_valid_wdnet_ids)))
        else:
            save_embd_sim_txt_path = os.path.join(root_dir, "meta_imgnet_sim_{}_top5.txt".format(model_name))
            save_embd_sim_json_path = os.path.join(root_dir, "meta_imgnet_sim_{}_top5.json".format(model_name))
        embd_sim = {}
        with open(save_embd_sim_txt_path, "w") as fw:
            for wdnet_id in tqdm(list(tfrecord_by_wdnet.keys())):
                if isG500:
                    if not (wdnet_id in g500_valid_wdnet_ids):
                        continue
                if not (wdnet_id in embd_sim):
                    embd_sim[wdnet_id] = []
                tfrecord_pathlist = tfrecord_by_wdnet[wdnet_id]
                imgnet_label_id = imgnet_embd_wdnet2id[wdnet_id]
                for tfrecord_path in tfrecord_pathlist:
                    embd_tf = web_embd[web_embd_tf2id[tfrecord_path]]
                    sim_cos = np.matmul(embd_tf, imgnet_embd)
                    ### sort the cosine similarity from maximum to minimum
                    sim_cos_argmax = np.argsort(sim_cos).tolist()[::-1]
                    assert(len(sim_cos) == 1000)
                    sim_cos_top_ids = [imgnet_label_id]
                    for sim_cos_top_id in sim_cos_argmax:
                        if isG500:
                            sim_cos_top_wdnet = imgnet_embd_id2wdnet[sim_cos_top_id]
                            if not (sim_cos_top_wdnet in g500_valid_wdnet_ids):
                                continue 
                            else:
                                if sim_cos_top_id in sim_cos_top_ids:
                                    continue
                                if len(sim_cos_top_ids) >= 5:
                                    break                      
                        else:
                            if sim_cos_top_id in sim_cos_top_ids:
                                continue
                            if len(sim_cos_top_ids) >= 5:
                                break
                        sim_cos_top_ids.append(sim_cos_top_id)
                    assert(len(sim_cos_top_ids) == 5)
                    text_results = [tfrecord_path]
                    for sim_cos_top_id in sim_cos_top_ids:
                        ### 计算top K类的所有相似度值
                        sim_cos_top_wdnet = imgnet_embd_id2wdnet[sim_cos_top_id]
                        if isG500:
                            assert (sim_cos_top_wdnet in g500_valid_wdnet_ids)
                        sim_cos_top = sim_cos[sim_cos_top_id]
                        text_results.append("%s@%.4f"%(str(sim_cos_top_wdnet),\
                        float(sim_cos_top)))
                    fw.write("\t".join(text_results) + "\n")
                    embd_sim[wdnet_id].append(text_results)
        with open(save_embd_sim_json_path, "w") as fw:
            json.dump(embd_sim, fw)
    return


def calculate_distance_kreciprocal(model_name, isTop5=False, isG500=False,
    is_smoothed=False, is_debug=False, is_rerank=False):
    """
    this function calculates the cosine similarity between each image's meta data and the class_name
    """
    root_dir = "dataset/webvision1k/meta_data_tf"
    tfrecord2wdnet_path = os.path.join(root_dir, "mapping_tfrecord2wdnet.txt")
    img_root_path = "dataset/webvision1k/tfrecord_webvision_train"
    if not os.path.exists(tfrecord2wdnet_path):
        print("Preprocess Mapping from TFRecord to WordNetID")
        img_index_path = os.path.join(img_root_path, "tfrecord_webvision_train.index")
        tfrecord_by_wdnet = {}
        with open(img_index_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split(" ")
                img_path = info[0]
                wdnet_id = str(info[1])
                tfrecord_name = info[2]
                tfrecord_offset = info[3]
                tfrecord_path = str(tfrecord_name) + "@" + str(tfrecord_offset)
                if not (wdnet_id in tfrecord_by_wdnet):
                    tfrecord_by_wdnet[wdnet_id] = []
                tfrecord_by_wdnet[wdnet_id].append(tfrecord_path)
        
        with open(tfrecord2wdnet_path, "w") as fw:
            for wdnet_id, tfrecord_pathlist in tfrecord_by_wdnet.items():
                for tfrecord_path in tfrecord_pathlist:
                    fw.write("\t".join([tfrecord_path, wdnet_id]) + "\n")
    else:
        print("Load Mapping from TFRecord to WordNetID")
        tfrecord_by_wdnet = {}
        with open(tfrecord2wdnet_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split("\t")
                tfrecord_path = info[0]
                wdnet_id = info[1]
                if not (wdnet_id in tfrecord_by_wdnet):
                    tfrecord_by_wdnet[wdnet_id] = []
                tfrecord_by_wdnet[wdnet_id].append(tfrecord_path)
    print("Load Mapping from Embd. to Index")
    web_embd_tf2id_path = os.path.join(root_dir, "mapping_tfrecord2index.txt")
    web_embd_tf2id = {}
    with open(web_embd_tf2id_path, "r") as fr:
        for line in fr.readlines():
            info = line.strip().split("\t")
            tfrecord_idx = info[0]
            tfrecord_path = info[1]
            web_embd_tf2id[tfrecord_path] = int(tfrecord_idx)

    imgnet_embd_wdnet2id_path = os.path.join(root_dir, "mapping_wdnet2index.txt")
    imgnet_embd_wdnet2id = {}
    imgnet_embd_id2wdnet = {}
    with open(imgnet_embd_wdnet2id_path, "r") as fr:
        for line in fr.readlines():
            wdnet_idx, wdnet_id = line.strip().split("\t")
            imgnet_embd_wdnet2id[wdnet_id] = int(wdnet_idx)
            imgnet_embd_id2wdnet[int(wdnet_idx)] = wdnet_id
    
    if is_smoothed:
        if is_rerank:
            smoothed_str = "knn_rerank_smoothed_" 
        else:
            smoothed_str = "knn_smoothed_"
    else:
        smoothed_str = ""

    web_embd_path = os.path.join(root_dir, "{}meta_embd_{}.npy".format(smoothed_str, model_name))
    imgnet_embd_path = os.path.join(root_dir, "imgnet_wdnet_embd_{}.npy".format(model_name))
    ## normalize along feature dimension
    print("Load embedding features-ing")
    web_embd = np.load(web_embd_path)
    imgnet_embd = np.load(imgnet_embd_path)
    web_embd_norm = np.linalg.norm(web_embd, axis=1, keepdims=True)
    imgnet_embd_norm = np.linalg.norm(imgnet_embd, axis=1, keepdims=True)
    web_embd = web_embd / web_embd_norm
    imgnet_embd = imgnet_embd / imgnet_embd_norm
    print("Start processing meta-imagenet distance")
    wdnet_ids = list(tfrecord_by_wdnet.keys())
    if is_debug:
        wdnet_ids = wdnet_ids[:5]

    if not isTop5:
        save_embd_dist_txt_path = os.path.join(root_dir, "{}meta_imgnet_dist_{}.txt".format(smoothed_str, model_name))
        save_embd_dist_json_path = os.path.join(root_dir, "{}meta_imgnet_dist_{}.json".format(smoothed_str, model_name))
        embd_dist = {}
        with open(save_embd_dist_txt_path, "w") as fw:
            for wdnet_id in tqdm(wdnet_ids):
                if not (wdnet_id in embd_dist):
                    embd_dist[wdnet_id] = []
                pos_id, offset_id = wdnet_id[0], wdnet_id[1:]
                syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
                print("wdnet id = {} names = {}".format(
                    wdnet_id,
                    syn.lemma_names(),
                    ))
                
                imgnet_embd_idx = imgnet_embd_wdnet2id[wdnet_id]
                embd_wdnet = imgnet_embd[imgnet_embd_idx:imgnet_embd_idx+1]  # probFeat
                tfrecord_pathlist = tfrecord_by_wdnet[wdnet_id]

                embd_tf_list = []
                for tfrecord_path in tfrecord_pathlist:
                    web_embd_idx = web_embd_tf2id[tfrecord_path]
                    embd_tf = web_embd[web_embd_idx:web_embd_idx+1]
                    embd_tf_list.append(embd_tf)
                embd_tf_list = np.concatenate(embd_tf_list, axis=0)  # galleryFeat
                
                dist_rerank = re_ranking(embd_wdnet, embd_tf_list,\
                    MemorySave=(embd_tf_list.shape[0] > 10000))[0]  # only one prob feature
                # idx_rerank = np.argsort(dist_rerank)
                for tfrecord_path, dist in zip(tfrecord_pathlist, dist_rerank):
                    fw.write("\t".join([tfrecord_path, wdnet_id, "%.4f"%(dist)]) + "\n")
                    embd_dist[wdnet_id].append([tfrecord_path, str(dist)])

        with open(save_embd_dist_json_path, "w") as fw:
            json.dump(embd_dist, fw)

    else:
        #### 计算top 4相似度的距离 & 对应原始网图的相似度距离
        if isG500:
            save_embd_dist_txt_path = os.path.join(root_dir,\
                "{}meta_imgnet_dist_{}_g500_top5.txt".format(smoothed_str, model_name))
            save_embd_dist_json_path = os.path.join(root_dir,\
                "{}meta_imgnet_dist_{}_g500_top5.json".format(smoothed_str, model_name))
            g500_valid_wdnet_ids = set()
            g500_valid_path = "filelist/mapping_google_500.txt"
            with open(g500_valid_path, "r") as fr:
                for line in fr.readlines():
                    wdnet_id = line.strip().split(" ")[1]
                    g500_valid_wdnet_ids.add(wdnet_id)
            assert(len(g500_valid_wdnet_ids) == 500), "number of valid wordnet ids = 500"
        else:
            save_embd_dist_txt_path = os.path.join(root_dir,\
                "{}meta_imgnet_dist_{}_top5.txt".format(smoothed_str, model_name))
            save_embd_dist_json_path = os.path.join(root_dir,\
                "{}meta_imgnet_dist_{}_top5.json".format(smoothed_str, model_name))

        embd_dist = {}
        with open(save_embd_dist_txt_path, "w") as fw:
            for wdnet_id in tqdm(wdnet_ids):
                if isG500:
                    if not (wdnet_id in g500_valid_wdnet_ids):
                        continue
                if not (wdnet_id in embd_dist):
                    embd_dist[wdnet_id] = []
                
                imgnet_label_id = imgnet_embd_wdnet2id[wdnet_id]
                embd_wdnet = imgnet_embd[imgnet_label_id:imgnet_label_id+1]  # probFeat
                tfrecord_pathlist = tfrecord_by_wdnet[wdnet_id]

                all_info = []
                embd_tf_list = []
                for tfrecord_path in tfrecord_pathlist:
                    web_embd_idx = web_embd_tf2id[tfrecord_path]
                    embd_tf = web_embd[web_embd_idx:web_embd_idx+1]
                    embd_tf_list.append(embd_tf)
                    sim_cos = np.matmul(web_embd[web_embd_idx], imgnet_embd.T)
                    ### sort the cosine similarity from maximum to minimum
                    sim_cos_argmax = np.argsort(sim_cos).tolist()[::-1]
                    assert(len(sim_cos) == 1000)
                    sim_cos_top_ids = [imgnet_label_id]
                    for sim_cos_top_id in sim_cos_argmax:
                        if isG500:
                            sim_cos_top_wdnet = imgnet_embd_id2wdnet[sim_cos_top_id]
                            if not (sim_cos_top_wdnet in g500_valid_wdnet_ids):
                                continue 
                            else:
                                if sim_cos_top_id in sim_cos_top_ids:
                                    continue
                                if len(sim_cos_top_ids) >= 5:
                                    break                      
                        else:
                            if sim_cos_top_id in sim_cos_top_ids:
                                continue
                            if len(sim_cos_top_ids) >= 5:
                                break
                        sim_cos_top_ids.append(sim_cos_top_id)
                    assert(len(sim_cos_top_ids) == 5)
                    text_results = [tfrecord_path]
                    for sim_cos_top_id in sim_cos_top_ids:
                        ### 计算top K类的所有相似度值
                        sim_cos_top_wdnet = imgnet_embd_id2wdnet[sim_cos_top_id]
                        if isG500:
                            assert (sim_cos_top_wdnet in g500_valid_wdnet_ids)
                        sim_cos_top = sim_cos[sim_cos_top_id]
                        text_results.append("%s@%.4f"%(str(sim_cos_top_wdnet),\
                        float(sim_cos_top)))
                    all_info.append(text_results)
                
                embd_tf_list = np.concatenate(embd_tf_list, axis=0)  # galleryFeat
                dist_rerank = re_ranking(embd_wdnet, embd_tf_list,\
                    MemorySave=(embd_tf_list.shape[0] > 10000))[0]  # only one prob feature
                # idx_rerank = np.argsort(dist_rerank)
                assert(len(dist_rerank) == len(all_info))
                
                for text_results, dist in zip(all_info, dist_rerank):
                    text_results_dist = text_results + ["%s@%.4f"%(str(wdnet_id),\
                        float(dist))]

                    fw.write("\t".join(text_results_dist) + "\n")
                    embd_dist[wdnet_id].append(text_results_dist)

        with open(save_embd_dist_json_path, "w") as fw:
            json.dump(embd_dist, fw)

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


def check_non_blanking_text():
    """过滤非空文本的图像
    """
    meta_json_path = "dataset/webvision1k/meta_data_tf/meta_data.json"
    save_root = "dataset/webvision1k/meta_data_tf"
    save_json_path = os.path.join(save_root, "meta_data_isValid.json")
    with open(meta_json_path, "r") as fr:
        meta_json = json.load(fr)
    with open(save_json_path, "w") as fw:
        isValid = {}
        for tfrecord_path in tqdm(meta_json.keys()):
            if meta_json[tfrecord_path]['text'] != "":
                isValid[tfrecord_path] = True
            else:
                isValid[tfrecord_path] = False
        json.dump(isValid, fw)
    return


def extract_top_k_by_cosine_similarity(model_name, root_dir,\
    save_root, tfrecord_root, ignore_none=False, isTop5=False,\
    is_dist=False, is_smoothed=False, is_rerank=False):
    """extract top k samples by the cosine similarity between meta info and class_text
    model_name: the name of model to encode sentences
    save_root: the directory path to save images
    ignore_none: ignore blanking items in the text
    tfrecord_root: the directory path to load tfrecord images
    """
    timestr = time.strftime("%m%d_%H%M%S")
    suffix_root = ["vis"]
    if ignore_none:
        suffix_root.append("isValid")
    if isTop5:
        suffix_root.append("isTop5")
    suffix_root = "_".join(suffix_root)
    save_root = os.path.join(save_root, model_name + "_{}_{}".format(suffix_root, timestr))
    os.makedirs(save_root, exist_ok=True)

    if is_smoothed:
        if is_rerank:
            smoothed_str = "knn_rerank_smoothed_" 
        else:
            smoothed_str = "knn_smoothed_"
    else:
        smoothed_str = ""

    if isTop5:
        if is_dist:
            embd_sim_json_path = os.path.join(root_dir,\
                "{}meta_imgnet_dist_{}_top5.json".format(smoothed_str, model_name))
        else:
            embd_sim_json_path = os.path.join(root_dir,\
                "{}meta_imgnet_sim_{}_top5.json".format(smoothed_str, model_name))
    else:
        if is_dist:
            embd_sim_json_path = os.path.join(root_dir,\
                "{}meta_imgnet_dist_{}.json".format(smoothed_str, model_name))
        else:
            embd_sim_json_path = os.path.join(root_dir,\
                "{}meta_imgnet_sim_{}.json".format(smoothed_str, model_name))

    print("embd_sim_json_path: ", embd_sim_json_path)
    assert(os.path.exists(embd_sim_json_path))
    with open(embd_sim_json_path, "r") as fr:
        embd_sim = json.load(fr)
    if ignore_none:
        isValid_json_path = os.path.join(root_dir, "meta_data_isValid.json")
        with open(isValid_json_path, "r") as fr:
            isValid = json.load(fr)

    ambiguity_wdnet_ids = ["n03250847", "n03804744", "n02012849",\
    "n03126707", "n02123159", "n13133613", "n02229544", "n02231487",\
    "n02277742", "n03710721", "n03710637", "n04296562", "n03532672",\
    "n02124075", "n03793489", "n04286575", "n03633091", "n04560804",\
    "n04264628", "n04277352", "n03933933"]

    wdnet_ids = set([ambiguity_wdnet_id for ambiguity_wdnet_id in ambiguity_wdnet_ids if ambiguity_wdnet_id in embd_sim] +\
        random.sample(list(embd_sim.keys()), 5))
    
    for wdnet_id in tqdm(wdnet_ids):
        pos_id, offset_id = wdnet_id[0], wdnet_id[1:]
        syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
        save_dir_name = wdnet_id + "_" + "_".join(syn.lemma_names()[:2]).replace("-", "_").replace(" ", "_")
        save_dir_path = os.path.join(save_root, save_dir_name)
        os.makedirs(save_dir_path, exist_ok=True)
        tfrecord_pathlist = embd_sim[wdnet_id]
        if isTop5:
            ### split the text
            if is_dist:
                for idx in range(len(tfrecord_pathlist)):
                    tfrecord_pathlist[idx][6] = float(tfrecord_pathlist[idx][6].split("@")[1])
                tfrecord_pathlist = sorted(tfrecord_pathlist, key=lambda x:x[6], reverse=False)
            else:
                for idx in range(len(tfrecord_pathlist)):
                    tfrecord_pathlist[idx][1] = float(tfrecord_pathlist[idx][1].split("@")[1])
                tfrecord_pathlist = sorted(tfrecord_pathlist, key=lambda x:x[1], reverse=True)
        else:
            for idx in range(len(tfrecord_pathlist)):
                tfrecord_pathlist[idx][1] = float(tfrecord_pathlist[idx][1])
            if is_dist:
                tfrecord_pathlist = sorted(tfrecord_pathlist, key=lambda x:x[1], reverse=False)
            else:
                tfrecord_pathlist = sorted(tfrecord_pathlist, key=lambda x:x[1], reverse=True)

        if ignore_none:
            tfrecord_pathlist = [tfrecord_item for tfrecord_item in tfrecord_pathlist if isValid[tfrecord_item[0]]]
        tfrecord_pathlist_top_sim = tfrecord_pathlist[:50]
        tfrecord_pathlist_btm_sim = tfrecord_pathlist[-50:]
        for tfrecord_item in tfrecord_pathlist_top_sim:
            tfrecord_path = tfrecord_item[0]
            tfrecord_name, tfrecord_offset = tfrecord_path.strip().split("@")
            tfrecord_name_path = os.path.join(tfrecord_root, tfrecord_name)
            assert(os.path.exists(tfrecord_name_path))
            img = get_tfrecord_image(tfrecord_name_path, int(tfrecord_offset))
            if is_dist:
                sim_cos = tfrecord_item[6]
            else:
                sim_cos = tfrecord_item[1]
            if isTop5:
                text_top4 = []
                for text_sim in tfrecord_item[2:6]:
                    cos_wdnet_id, cos_sim = text_sim.split("@")
                    cos_sim = float(cos_sim)
                    pos_id, offset_id = cos_wdnet_id[0], cos_wdnet_id[1:]
                    syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
                    syn_name = "_".join(syn.lemma_names()[:2])
                    text_top4.append("%s_%.2f"%(syn_name, cos_sim))
                text_top4 = "_".join(text_top4)
                save_img_path = os.path.join(save_dir_path, "top_%.2f_%s_%s.jpg"%(sim_cos, text_top4, str(tfrecord_path)))
            else:
                save_img_path = os.path.join(save_dir_path, "top_%.2f_%s.jpg"%(sim_cos, str(tfrecord_path)))
            img.save(save_img_path)
        for tfrecord_item in tfrecord_pathlist_btm_sim:
            tfrecord_path = tfrecord_item[0]
            tfrecord_name, tfrecord_offset = tfrecord_path.strip().split("@")
            tfrecord_name_path = os.path.join(tfrecord_root, tfrecord_name)
            assert(os.path.exists(tfrecord_name_path))
            img = get_tfrecord_image(tfrecord_name_path, int(tfrecord_offset))
            if is_dist:
                sim_cos = tfrecord_item[6]
            else:
                sim_cos = tfrecord_item[1]
            if isTop5:
                text_top4 = []
                for text_sim in tfrecord_item[2:6]:
                    cos_wdnet_id, cos_sim = text_sim.split("@")
                    cos_sim = float(cos_sim)
                    pos_id, offset_id = cos_wdnet_id[0], cos_wdnet_id[1:]
                    syn = wn.synset_from_pos_and_offset(pos_id, int(offset_id))
                    syn_name = "_".join(syn.lemma_names()[:2])
                    text_top4.append("%s_%.2f"%(syn_name, cos_sim))
                text_top4 = "_".join(text_top4)
                save_img_path = os.path.join(save_dir_path, "btm_%.2f_%s_%s.jpg"%(sim_cos, text_top4, str(tfrecord_path)))
            else:
                save_img_path = os.path.join(save_dir_path, "btm_%.2f_%s.jpg"%(sim_cos, str(tfrecord_path)))
            img.save(save_img_path)
    return


def add_meta_info(model_name, isG500=False, isKEEP=False, is_smoothed=False, is_rerank=False, is_dist=False):
    """
    将所有的样本增加对应的类别编号下的meta文本匹配相似度
    isG500: 使用g500数据集
    isKEEP: 使用简单策略过滤一遍数据
    """
    root_dir = "dataset/webvision1k/meta_data_tf"
    if is_smoothed:
        if is_rerank:
            smoothed_str = "knn_rerank_smoothed_" 
        else:
            smoothed_str = "knn_smoothed_"
    else:
        smoothed_str = ""

    if isG500:
        if is_dist:
            embd_sim_txt_path = os.path.join(root_dir, "{}meta_imgnet_dist_{}_g500_top5.txt".format(smoothed_str, model_name)) 
        else:
            embd_sim_txt_path = os.path.join(root_dir, "{}meta_imgnet_sim_{}_g500_top5.txt".format(smoothed_str, model_name))
        mapping_wdnet2id_path = "filelist/mapping_google_500.txt"
        if isKEEP:
            train_filelist_path = "filelist/train_filelist_google_500_usable_tf_keep.txt"
        else:
            train_filelist_path = "filelist/train_filelist_google_500_usable_tf.txt"
    else:
        if is_dist:
            embd_sim_txt_path = os.path.join(root_dir, "{}meta_imgnet_dist_{}_top5.txt".format(smoothed_str, model_name))
        else:
            embd_sim_txt_path = os.path.join(root_dir, "{}meta_imgnet_sim_{}_top5.txt".format(smoothed_str, model_name))
        mapping_wdnet2id_path = "filelist/mapping_webvision_1k.txt"
        if isKEEP:
            train_filelist_path = "filelist/train_filelist_webvision_1k_usable_tf_keep.txt"
        else:
            train_filelist_path = "filelist/train_filelist_webvision_1k_usable_tf.txt"

    print("embd_sim_txt_path:", embd_sim_txt_path)
    assert(os.path.exists(embd_sim_txt_path))

    print("train_filelist_path:", train_filelist_path)
    assert(os.path.exists(train_filelist_path))

    mapping_wdnet2id = {}
    with open(mapping_wdnet2id_path, "r") as fr:
        for line in fr.readlines():
            info = line.strip().split(" ")
            wdnet_id, class_id = info[1], int(info[0])
            mapping_wdnet2id[wdnet_id] = class_id
            # print("wordnet id: {}".format(wdnet_id))

    embd_sim = {}
    with open(embd_sim_txt_path, "r") as fr:
        for line in tqdm(fr.readlines()):
            info = line.strip().split("\t")
            tfrecord_path = info[0]
            meta_class_ids = []
            meta_confs = []
            for meta_info in info[1:6]:
                ### 计算各个类别下的匹配程度
                wdnet_id, conf_id = meta_info.strip().split("@")
                conf_id = float(conf_id)
                class_id = mapping_wdnet2id[wdnet_id]
                meta_class_ids.append(class_id)
                meta_confs.append(conf_id)
            if is_dist:
                meta_dist = float(info[6].split("@")[1])
                embd_sim[tfrecord_path] = [meta_class_ids, meta_confs, meta_dist]
            else:
                embd_sim[tfrecord_path] = [meta_class_ids, meta_confs]

    print("Start writing text files")
    save_txt_path = train_filelist_path.replace(".txt", "_{}meta_{}.txt".format(smoothed_str, model_name))

    assert(save_txt_path != train_filelist_path)
    with open(train_filelist_path, "r") as fr:
        with open(save_txt_path, "w") as fw:
            for line in tqdm(fr.readlines()):
                info = line.strip().split(" ")
                tfrecord_path = info[0]
                info_json_str = " ".join(info[1:])
                info_json = json.loads(info_json_str)
                meta_confs = embd_sim[tfrecord_path]
                info_json["meta_label"] = meta_confs[0]
                info_json["meta_sim"] = meta_confs[1]
                if is_dist:
                    info_json["meta_dist"] = meta_confs[2]
                info_json_str = json.dumps(info_json)
                fw.write(" ".join([tfrecord_path, info_json_str]) + "\n")

    return


if __name__ == "__main__":
    ## 预处理meta文本信息
    preprocess()
    
    ## 抽取文本embeddings
    extract_embeddings(model_name="minilm", is_meta=True)
    extract_embeddings(model_name="xlnet", is_meta=True)
    extract_embeddings(model_name="minilm", is_meta=False)
    extract_embeddings(model_name="xlnet", is_meta=False)
    extract_embeddings(model_name="gpt", is_meta=False)
    extract_embeddings(model_name="gpt", is_meta=True)

    # calculate_cosine_similarity("minilm", isTop5=True)
    # calculate_cosine_similarity("xlnet", isTop5=True)
    # calculate_cosine_similarity("minilm", isTop5=True, isG500=True)
    # calculate_cosine_similarity("xlnet", isTop5=True, isG500=True)

    ## 按照ReRank思路抽取每个类别样本的标签
    # ###=======================================================###
    root_dir = "dataset/webvision1k/meta_data_tf"
    # tfrecord_root = "dataset/webvision1k/tfrecord_webvision_train"
    # vis_root = "dataset/webvision1k/visualize_meta_data_dist"
    # os.makedirs(vis_root, exist_ok=True)

    print("========================================================")
    calculate_distance_kreciprocal("minilm", isTop5=True, isG500=False, is_smoothed=True, is_debug=False)
    # save_root = os.path.join(vis_root, "meta_data_visualize_KNN-smooth_minilm")
    # extract_top_k_by_cosine_similarity("minilm", root_dir, save_root, tfrecord_root, ignore_none=False, isTop5=True, is_dist=True, is_smoothed=True)
    print("========================================================")
    
    print("========================================================")
    calculate_distance_kreciprocal("xlnet", isTop5=True, isG500=False, is_smoothed=True, is_debug=False)
    # save_root = os.path.join(vis_root, "meta_data_visualize_KNN-smooth_xlnet")
    # extract_top_k_by_cosine_similarity("minilm", root_dir, save_root, tfrecord_root,
    #    ignore_none=False, isTop5=True, is_dist=True, is_smoothed=True)
    print("========================================================")

    print("========================================================")
    calculate_distance_kreciprocal("gpt", isTop5=True, isG500=False, is_smoothed=True, is_debug=False)
    print("========================================================")
    calculate_distance_kreciprocal("minilm", isTop5=True, isG500=True, is_smoothed=True, is_debug=False)
    print("========================================================")
    calculate_distance_kreciprocal("xlnet", isTop5=True, isG500=True, is_smoothed=True, is_debug=False)
    print("========================================================")

    print("========================================================")
    calculate_distance_kreciprocal("gpt", isTop5=True, isG500=True, is_smoothed=True, is_debug=False)
    print("========================================================")

    # ##########=======================================================###
    # vis_root = "dataset/webvision1k/visualize_meta_data_dist_rerank"
    # os.makedirs(vis_root, exist_ok=True)

    print("========================================================")
    calculate_distance_kreciprocal("minilm", isTop5=True, isG500=False, is_smoothed=True, is_debug=False, is_rerank=True)
    # save_root = os.path.join(vis_root, "meta_data_visualize_KNN-smooth_minilm")
    # extract_top_k_by_cosine_similarity("minilm", root_dir, save_root, tfrecord_root, ignore_none=False, isTop5=True, is_dist=True, is_rerank=True, is_smoothed=True)
    print("========================================================")

    print("========================================================")
    calculate_distance_kreciprocal("gpt", isTop5=True, isG500=False, is_smoothed=True, is_debug=False, is_rerank=True)
    print("========================================================")

    print("========================================================")
    calculate_distance_kreciprocal("xlnet", isTop5=True, isG500=False, is_smoothed=True, is_debug=False, is_rerank=True)
    # save_root = os.path.join(vis_root, "meta_data_visualize_KNN-smooth_xlnet")
    # extract_top_k_by_cosine_similarity("minilm", root_dir, save_root, tfrecord_root, ignore_none=False, isTop5=True, is_dist=True, is_rerank=True, is_smoothed=True)
    print("========================================================")

    print("========================================================")
    calculate_distance_kreciprocal("gpt", isTop5=True, isG500=True, is_smoothed=True, is_debug=False, is_rerank=True)
    print("========================================================")

    print("========================================================")
    calculate_distance_kreciprocal("minilm", isTop5=True, isG500=True, is_smoothed=True, is_debug=False, is_rerank=True)
    print("========================================================")

    print("========================================================")
    calculate_distance_kreciprocal("xlnet", isTop5=True, isG500=True, is_smoothed=True, is_debug=False, is_rerank=True)
    print("========================================================")

    add_meta_info("minilm", isG500=False, isKEEP=False, is_smoothed=True, is_rerank=False, is_dist=True)
    add_meta_info("minilm", isG500=True, isKEEP=False, is_smoothed=True, is_rerank=False, is_dist=True)
    add_meta_info("xlnet", isG500=False, isKEEP=False, is_smoothed=True, is_rerank=False, is_dist=True)
    add_meta_info("xlnet", isG500=True, isKEEP=False, is_smoothed=True, is_rerank=False, is_dist=True)
    add_meta_info("gpt", isG500=False, isKEEP=False, is_smoothed=True, is_rerank=False, is_dist=True)
    add_meta_info("gpt", isG500=True, isKEEP=False, is_smoothed=True, is_rerank=False, is_dist=True)

    add_meta_info("minilm", isG500=False, isKEEP=False, is_smoothed=True, is_rerank=True, is_dist=True)
    add_meta_info("minilm", isG500=True, isKEEP=False, is_smoothed=True, is_rerank=True, is_dist=True)
    add_meta_info("xlnet", isG500=False, isKEEP=False, is_smoothed=True, is_rerank=True, is_dist=True)
    add_meta_info("xlnet", isG500=True, isKEEP=False, is_smoothed=True, is_rerank=True, is_dist=True)
    add_meta_info("gpt", isG500=False, isKEEP=False, is_smoothed=True, is_rerank=True, is_dist=True)
    add_meta_info("gpt", isG500=True, isKEEP=False, is_smoothed=True, is_rerank=True, is_dist=True)


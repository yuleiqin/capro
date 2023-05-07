import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from .knn_utils import global_build, sgc_precompute


def load_nuswide(filelist_path, num_class, is_eval=False):
    tfrecord_by_class = {}
    tfrecord_pathlist = []
    tfrecord_labels = []
    with open(filelist_path, "r") as fr:
        for line in tqdm(fr.readlines()):
            tfrecord_offset = line.strip().split(" ")[0]
            tfrecord_pathlist.append(tfrecord_offset)
            targets_multihot = np.zeros((num_class,))
            meta_info = line.strip().replace(tfrecord_offset + " ", "")
            if is_eval:
                targets = meta_info.split(",")
                targets = [int(target) for target in targets]
                for target in targets:
                    targets_multihot[target] = 1
                if not (target in tfrecord_by_class):
                    tfrecord_by_class[target] = []
                tfrecord_by_class[target].append([tfrecord_offset, 1.0, 0.0])
            else:
                json_info = json.loads(meta_info)
                targets = [int(target) for target in json_info["meta_label"]]
                target_sims = [float(sim) for sim in json_info["meta_sim"]]
                target_dists = [float(dist) for dist in json_info["meta_dist"]]
                ### 默认拆解成多类sigmoid
                for target, target_sim, target_dist in zip(targets,
                    target_sims, target_dists):
                    targets_multihot[target] = 1
                    if not (target in tfrecord_by_class):
                        tfrecord_by_class[target] = []
                    tfrecord_by_class[target].append([tfrecord_offset, target_sim, target_dist])
            tfrecord_labels.append(targets_multihot)
    tfrecord_labels = np.array(tfrecord_labels)
    return tfrecord_by_class, tfrecord_pathlist, tfrecord_labels


def load_webvision(filelist_path, num_class, is_eval=False):
    tfrecord_by_class = {}
    tfrecord_pathlist = []
    tfrecord_labels = []
    with open(filelist_path, "r") as fr:
        for line in fr.readlines():
            tfrecord_offset = line.strip().split(" ")[0]
            tfrecord_pathlist.append(tfrecord_offset)
            meta_info = line.strip().replace(tfrecord_offset + " ", "")
            targets_multihot = np.zeros((num_class,))
            if is_eval:
                target = int(meta_info)
                targets_multihot[target] = 1
                target_sim = 1
                target_dist = 0
            else:
                json_info = json.loads(meta_info)
                target = int(json_info["meta_label"][0])
                targets_multihot[target] = 1
                target_sim = float(json_info["meta_sim"][0])
                target_dist = float(json_info["meta_dist"])
            tfrecord_labels.append(targets_multihot)
            if not (target in tfrecord_by_class):
                tfrecord_by_class[target] = []
            tfrecord_by_class[target].append([tfrecord_offset, target_sim, target_dist])
    tfrecord_labels = np.array(tfrecord_labels)
    return tfrecord_by_class, tfrecord_pathlist, tfrecord_labels


def load_filelist(filelist_path, nuswide, num_class, is_eval=False):
    if nuswide:
        return load_nuswide(filelist_path, num_class, is_eval)
    else:
        return load_webvision(filelist_path, num_class, is_eval)


def load_mapping_tfrecord2embd_idx(mapping_tfrecord2embd_idx_path):
    if mapping_tfrecord2embd_idx_path.endswith("json"):
        with open(mapping_tfrecord2embd_idx_path, "r") as fr:
            tf2embd_id = json.load(fr)
    else:
        tf2embd_id = {}
        with open(mapping_tfrecord2embd_idx_path, "r") as fr:
            for line in fr.readlines():
                info = line.strip().split("\t")
                tfrecord_path = info[0]
                embd_idx = int(info[1])
                tf2embd_id[tfrecord_path] = embd_idx
    return tf2embd_id


def get_default_sample_weight(label_train):
    print("==============>Use Default Weights Reweighting<==============")
    num_class = label_train.shape[1]
    class_nums_pos = [1.0 for _ in range(num_class)]
    class_nums_neg = [1.0 for _ in range(num_class)]
    label_class_ids = []
    for label_i in label_train:
        label_class_id = []
        for idx in range(len(label_i)):
            if int(label_i[idx]) == 1:
                class_nums_pos[idx] += 1
                label_class_id.append(idx)
            elif int(label_i[idx]) == 0:
                class_nums_neg[idx] += 1
        label_class_ids.append(label_class_id)

    class_nums_pos = np.array(class_nums_pos)
    class_nums_neg = np.array(class_nums_neg)
    class_median_pos = np.median(class_nums_pos)
    class_weight = class_nums_pos / class_median_pos

    sample_weight = []
    for label_class_id in label_class_ids:
        if len(label_class_id) > 0:
            sample_weight.append(max([class_weight[idx] for idx in label_class_id]))
        else:
            sample_weight.append(1.0)
    sample_weight = np.array(sample_weight)
    pos_weight = np.sqrt(class_nums_neg/class_nums_pos)
    return sample_weight, pos_weight


def get_label_onehot(tfrecord_by_class, num_class, nuswide=False,
    text_topk=(50, 50), text_conf=(0.1, 0.1), text_prop=(1./3, 1./3), use_rerank=False): 
    targets = list(tfrecord_by_class.keys())
    assert(num_class == max(targets) + 1)
    assert(num_class == len(targets))
    label_onehot_by_tfrecord = {}
    num_samples_by_class = []
    for target in tqdm(targets):
        img_list = tfrecord_by_class[target]
        ### [tfrecord_path, similarity, rerank distance]
        if use_rerank:
            ## distance from low to high (sort wo reverse)
            img_list_sorted = sorted(img_list, key=lambda x:x[2], reverse=False)
        else:
            ## similarity from high to low (sort w reverse)
            img_list_sorted = sorted(img_list, key=lambda x:x[1], reverse=True)

        ### 筛选正负样本
        ### 按照位次排序固定数目
        if not (text_topk is None):
            num_positive_topk, num_negative_topk = text_topk
        else:
            num_positive_topk, num_negative_topk = 0, 0
        ### 按照阈值固定数目
        if not (text_conf is None):
            if use_rerank:
                num_positive_conf = sum([1 for img_item in img_list_sorted if 1-img_item[2] >= text_conf[0]])
                num_negative_conf = sum([1 for img_item in img_list_sorted if 1-img_item[2] < text_conf[1]])
            else:
                num_positive_conf = sum([1 for img_item in img_list_sorted if img_item[1] >= text_conf[0]])
                num_negative_conf = sum([1 for img_item in img_list_sorted if img_item[1] < text_conf[0]])
        else:
            num_positive_conf, num_negative_conf = 0, 0
        ### 按照比例数目
        if not (text_prop is None):
            num_positive_prop = int(len(img_list)*text_prop[0])
            num_negative_prop = int(len(img_list)*text_prop[1])
        else:
            num_positive_prop, num_negative_prop = 0, 0
        if len(img_list) > 50:
            buffer_negative = 50
        else:
            buffer_negative = 0
        num_positive = int(max(1, min(max((num_positive_topk, num_positive_conf, num_positive_prop)), len(img_list)-buffer_negative)))
        num_negative = int(min(max((num_negative_topk, num_negative_conf, num_negative_prop, buffer_negative)), len(img_list)-num_positive))
        num_samples_by_class.append([target, num_positive, num_negative, num_positive+num_negative])
        for img_item in img_list_sorted[:num_positive]:
            tfrecord_path = img_item[0]
            if nuswide:
                ## 多标签数据集只能确定该样本当前类别是1 其他类别为-1(未知)
                if not (tfrecord_path in label_onehot_by_tfrecord):
                    label_onehot_by_tfrecord[tfrecord_path] = np.ones((num_class,))*(-1)
                label_onehot_by_tfrecord[tfrecord_path][target] = 1
            else:
                ## 单标签数据集具有排他性
                one_hot = np.zeros((num_class,))
                one_hot[target] = 1
                label_onehot_by_tfrecord[tfrecord_path] = one_hot
        if num_negative > 1:
            for img_item in img_list_sorted[-num_negative:]:
                tfrecord_path = img_item[0]
                if nuswide:
                    ## 多标签数据集只能确定该样本当前类别是0 其他类别为-1(未知)
                    if not (tfrecord_path in label_onehot_by_tfrecord):
                        label_onehot_by_tfrecord[tfrecord_path] = np.ones((num_class,))*(-1)
                    label_onehot_by_tfrecord[tfrecord_path][target] = 0
                else:
                    one_hot = np.ones((num_class,))*(-1)
                    one_hot[target] = 0
                    ## 负样本也只能知道并不属于该类别
                    label_onehot_by_tfrecord[tfrecord_path] = one_hot
    tfrecord_pathlist = set(list(label_onehot_by_tfrecord.keys()))
    num_samples_by_class = sorted(num_samples_by_class, key=lambda x:x[3])
    num_positives_total = sum([x[1] for x in num_samples_by_class])
    num_negatives_total = sum([x[2] for x in num_samples_by_class])
    print("number of tfrecord samples for training minimum", num_samples_by_class[0])
    print("number of tfrecord samples for training maximum", num_samples_by_class[-1])
    print("number of tfrecord samples positive", num_positives_total)
    print("number of tfrecord samples negative", num_negatives_total)
    return tfrecord_pathlist, label_onehot_by_tfrecord


def load_knn_graph(knn_path, num_neighbor_knn, use_rerank=False):
    eps = 1e-3
    knns = []
    if type(knn_path) is str and os.path.exists(knn_path):
        knns_np = np.load(knn_path)["knns"]
    else:
        knns_np = knn_path
    for knn in knns_np:
        ners = knn[0][:num_neighbor_knn].astype(np.int32)
        sims = knn[1][:num_neighbor_knn].astype(np.float32)
        sims_clip = []
        for sim in sims:
            if np.isnan(sim):
                sims_clip.append(1-eps)
            else:
                if sim >= 1-eps:
                    sims_clip.append(1-eps)
                elif sim <= eps:
                    sims_clip.append(eps)
                else:
                    sims_clip.append(sim)
        if use_rerank:
            sims_clip = [1.-sim for sim in sims_clip]
        sims_clip = np.array(sims_clip).astype(np.float32)
        knns.append((ners, sims_clip))
    return knns


def getlabelarray(imglist_path):
    img_name = []
    label_set = []
    with open(imglist_path, 'r') as reader:
        for line in reader.readlines():
            words = re.split(',| |{|}|"|:|\n', line)
            words = list(filter(None, words))
            img_name.append(words[0])
            if 'label' in words:
                label = words[words.index('label') + 1]
            else:
                label = words[1]
            label_set.append(int(label))
    img_name = np.array(img_name)
    label_set = np.array(label_set)
    return img_name, label_set


def create_df(imglist_path, df_file_path, feature_file=None):
    [img_name, label] = getlabelarray(imglist_path)
    df = pd.DataFrame()
    df['img_name'] = img_name
    df['y'] = label
    if feature_file is not None:
        df = add_tsne(df, feature_file)
    df.to_pickle(df_file_path)
    return df


def add_pseudo(df, pseudo_path, select_idx, class_set):
    img_name, label = getlabelarray(pseudo_path)
    plabel = label[select_idx]
    df['pseudo'] = plabel
    for idx, pseudo_label in enumerate(plabel):
        if pseudo_label not in class_set:
            plabel[idx] = 1000
    df['yp'] = plabel
    return df


def add_tsne(df, feature_file):
    x = np.load(feature_file)
    assert len(x) == len(df); print('Df number does not match feature number')
    df = tsne_compute(df, x)
    return df


def tsne_compute(df, x):
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(x)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    df['tsne-pca50-one'] = tsne_pca_results[:, 0]
    df['tsne-pca50-two'] = tsne_pca_results[:, 1]
    return df


def get_conf(df, text_path, label_path, imglist_path, class_num, topk=5):
    text_feature = np.load(text_path)
    label_description = np.load(label_path)
    label_set = np.array(df['y'])
    conf_list = []
    conf_mark_list = []
    for i in class_num:
        class_text = text_feature[np.where(label_set == i)]
        conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
        conf_mark = 10000 * np.ones(conf.shape)
        conf_mark[np.argsort(-conf)[:topk]] = i
        conf_mark_list.extend(conf_mark)
        conf_list.extend(conf)
    conf_mark_list = np.array(conf_mark_list).astype(np.int)
    conf_list = np.array(conf_list)
    df['conf'] = conf_list
    df['seed'] = conf_mark_list
    return df


# def get_knn_conf(df, text_path, visual_path, label_path, imglist_path, class_num, save_filename, dist_def, k=5, topk=5,
#                  self_weight=0, edge_weight=True, build_graph_method='gpu'):
#     text_feature = np.load(text_path)
#     knn_graph = global_build(visual_path, dist_def, k, save_filename, build_graph_method)
#     text_feature = sgc_precompute(text_feature, knn_graph, self_weight=self_weight, edge_weight=edge_weight, degree=1)
#     label_description = np.load(label_path)
#     label_set = np.array(df['y'])
#     conf_list = np.zeros(np.shape(label_set))
#     conf_mark_list = np.zeros(np.shape(label_set))
#     for i in class_num:
#         class_google_idx = np.array(df[df['y'] == i].index)
#         class_text = text_feature[class_google_idx]
#         conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
#         conf_mark = 10000 * np.ones(conf.shape)
#         conf_mark[np.argsort(-conf)[:topk]] = i
#         conf_list[class_google_idx] = conf
#         conf_mark_list[class_google_idx] = np.array(conf_mark).astype(np.int)
#         if i%100==0:
#             print('Seed selection class-{} completed.'.format(i), flush=True)
#     conf_mark_list = np.array(conf_mark_list).astype(np.int)
#     conf_list = np.array(conf_list)
#     df['conf'] = conf_list
#     df['seed'] = conf_mark_list
#     return df


# def get_knn_conf_multisource(df, text_path, visual_path, label_path, imglist_path, class_num, save_filename, dist_def, k=5, topk=5,
#                  self_weight=0, edge_weight=True, build_graph_method='gpu'):
#     text_feature = np.load(text_path)
#     knn_graph = global_build(visual_path, dist_def, k, save_filename, build_graph_method)
#     text_feature = sgc_precompute(text_feature, knn_graph, self_weight=self_weight, edge_weight=edge_weight, degree=1)
#     label_description = np.load(label_path)
#     label_set = np.array(df['y'])
#     conf_list = np.zeros(np.shape(label_set))
#     conf_mark_list = np.zeros(np.shape(label_set))
#     for i in class_num:
#         class_google_idx = np.array(df[df['img_name'].str.startswith('google') & (df['y']==i)].index)
#         class_text = text_feature[class_google_idx]
#         conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
#         conf_mark = 10000 * np.ones(conf.shape)
#         conf_mark[np.argsort(-conf)[:topk]] = i
#         conf_list[class_google_idx] = conf
#         conf_mark_list[class_google_idx] = np.array(conf_mark).astype(np.int)

#         class_flickr_idx = np.array(df[df['img_name'].str.startswith('flickr') & (df['y']==i)].index)
#         if len(class_flickr_idx) > 0:
#             class_text = text_feature[class_flickr_idx]
#             conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
#             conf_mark = 10000 * np.ones(conf.shape)
#             conf_mark[np.argsort(-conf)[:topk]] = i
#             conf_list[class_flickr_idx] = conf
#             conf_mark_list[class_flickr_idx] = np.array(conf_mark).astype(np.int)
#         else:
#             print('Class {} does not contain Flickr files.'.format(i),flush=True)
#         if i%100==0:
#             print('Seed selection class-{} completed.'.format(i),flush=True)
#     conf_mark_list = np.array(conf_mark_list).astype(np.int)
#     conf_list = np.array(conf_list)
#     df['conf'] = conf_list
#     df['seed'] = conf_mark_list
#     return df


def data_slicing(train_imglist_path, train_feature_path, train_graph_feature_path,
                 val_imglist_path, val_feature_path, val_graph_feature_path):

    train_imgname, train_label = getlabelarray(train_imglist_path)
    val_imgname, val_label = getlabelarray(val_imglist_path)

    train_feature = np.load(train_feature_path)
    val_feature = np.load(val_feature_path)

    train_graph_feature = np.load(train_graph_feature_path)
    val_graph_feature = np.load(val_graph_feature_path)

    imglist = np.concatenate((train_imgname, val_imgname), axis=0)
    feature = np.concatenate((train_feature, val_feature), axis=0)
    graph_feature = np.concatenate((train_graph_feature, val_graph_feature), axis=0)
    label = np.concatenate((train_label, val_label), axis=0)

    train_idx = np.arange(len(train_feature))
    val_idx = np.arange(len(val_feature)) + len(train_feature)

    return imglist, feature, graph_feature, label, train_idx, val_idx


def get_revised_imglist(train_imglist_path, pred_labels, revised_imglist_path):
    train_imgname, train_label = getlabelarray(train_imglist_path)
    assert len(train_imgname) == len(pred_labels), "train imglist should not equal to prediction list"
    sample_reweight = get_reweight_ratio(pred_labels)
    newlines = []
    for i, imgname in enumerate(train_imgname):
        target_line = '{} {}\n'.format(
            imgname,
            json.dumps({
                'label': int(pred_labels[i]),
                'sample_weight': sample_reweight[i],
            }, sort_keys=True)
        )
        newlines.append(target_line)
    with open(revised_imglist_path, 'w') as writer:
        writer.writelines(newlines)


def get_revised_filelist(filelist_path, predictions, revised_filelist_path,
    mapping_tfrecord2embd_idx, nuswide=False):
    if nuswide:
        pred_labels_top3, pred_labels, pred_logits = predictions
    else:
        pred_labels, pred_logits = predictions
    with open(revised_filelist_path, "w") as fw:
        with open(filelist_path, "r") as fr:
            for line in tqdm(fr.readlines()):
                tfrecord_path = line.strip().split(" ")[0]
                embd_idx = int(mapping_tfrecord2embd_idx[tfrecord_path])
                json_info_str = line.strip().replace(tfrecord_path + " ", "")
                info = json.loads(json_info_str)
                if nuswide:
                    info["label_gnn"] = pred_labels[embd_idx]
                    info["label_gnn_top3"] = (pred_labels_top3[embd_idx]).tolist()
                else:
                    pred_label = int(pred_labels[embd_idx])
                    pred_label_logit = float(pred_logits[embd_idx][pred_label])
                    info["label_gnn"] = [pred_label, pred_label_logit]
                json_info_str_new = json.dumps(info)
                fw.write("{} {}\n".format(tfrecord_path, json_info_str_new))
    return


def get_reweight_ratio(pred_labels):
    try:
        class_and_counts = np.array(sorted(Counter(pred_labels).items()))
        np.testing.assert_array_equal(
            class_and_counts[:, 0],
            np.arange(class_and_counts.shape[0]),
        )
        class_freq = class_and_counts[:, 1]
        class_weight = 1 / class_freq
        sample_weight = class_weight[pred_labels]
        sample_weight *= 1 / sample_weight.mean()
    except:
        print('Some classes are missing!!!', flush=True)
        sample_weight = np.ones(pred_labels.shape)
    return sample_weight


def get_reweight_ratio_multihot(pred_labels_multihot):
    num_class = pred_labels_multihot.shape[1]
    try:
        class_freq = np.ones(num_class)
        sample_class = []
        for pred_label in pred_labels_multihot:
            sample_class_i = []
            for idx, pred_label_idx in enumerate(pred_label):
                if int(pred_label_idx) == 1:
                    class_freq[idx] += 1
                    sample_class_i.append(idx)
            sample_class.append(sample_class_i)

        class_weight = 1 / class_freq
        sample_weight = []
        for sample_class_i in sample_class:
            if len(sample_class_i) > 0:
                sample_weight_i = 1 / min([class_freq[class_idx] for class_idx in sample_class_i])
            else:
                sample_weight_i = 1 / np.mean(class_freq)
            sample_weight.append(sample_weight_i)
        sample_weight = np.array(sample_weight)
        sample_weight *= 1 / sample_weight.mean()
    except:
        print('Some classes are missing!!!', flush=True)
        sample_weight = np.ones(pred_labels.shape[0])
    return sample_weight


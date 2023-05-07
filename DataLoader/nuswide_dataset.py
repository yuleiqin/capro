from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import os
import io
import albumentations as alb
import struct
import json
import cv2
from copy import deepcopy as copy
import sys
sys.path.append("./")
from .example_pb2 import Example
from .fancy_pca import FancyPCA
sys.path.append("../")
from utils.rotate import rotate_and_crop
from utils.augmentations import RandomBorder, RandomTranslate
from utils.augmentations import RandomTextOverlay, RandomStripesOverlay


def get_concat_h_resize(im1, im2, resample=Image.BILINEAR, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst


class nuswide_dataset(Dataset):
    def __init__(self, root_dir, pathlist, transform, mode, num_class,\
        transform_strong=None, root_dir_target="", pathlist_target="",\
            save_dir="", dry_run=False, use_fewshot=True, annotation="",\
                no_color_transform=False, fast_eval=False, rebalance_downsample=False,\
                    use_meta_weights=0, topk=50):
        ###### 初始化设置
        assert(os.path.exists(root_dir))
        self.root_dir = root_dir
        assert(os.path.isfile(pathlist))
        self.pathlist = pathlist
        self.transform = transform
        self.mode = mode
        self.num_class = num_class
        self.imgs_by_class = None
        self.transform_strong = transform_strong
        self.root_dir_target = root_dir_target
        self.pathlist_target = pathlist_target
        assert(os.path.exists(save_dir))
        self.save_dir = save_dir
        self.dry_run = dry_run
        self.use_fewshot = use_fewshot
        self.annotation = annotation
        self.set_alb_transform()
        self.no_color_transform = no_color_transform
        self.fast_eval = fast_eval
        self.rebalance_downsample = rebalance_downsample
        self.use_meta_weights = use_meta_weights
        self.topk = topk
        if self.annotation != "" and os.path.exists(self.annotation):
            #### 第一种情况: 直接根据annotation json文件读取所有样本编号以及修正后的标签
            self.load_by_json()
        else:
            #### 第二种情况: 读取训练样本(文本格式)并保存进内存
            #### 1) 训练; 2) 测试; 3) 构建fewshot
            if self.mode == "train" or \
                self.mode == "test" or \
                    (self.mode == "fewshot" and not self.use_fewshot):
                self.load_by_txt()

            #### 判断来自target(real-world) domain的样本构建fewshot或加入训练集样本
            #### 1) 训练; 2) 构建fewshot
            if self.mode != "test":
                self.load_target_by_txt()
        
        if self.use_meta_weights > 0:
            ### 仅仅筛选相似度大于零的正样本
            self.keep_sample_w_positive_similarity()
            ## 根据匹配的分数&类别样本数的反比进行综合加权
            self.get_scaled_sample_weight()
        else:
            ### 根据类别样本数的反比进行加权
            self.get_default_sample_weight()

        if self.mode == "train" and not self.use_fewshot:
            ### 构建伪标签pseudo-fewshot
            self.check_clean_sample_by_rank_distance()

        if self.dry_run:
            self._dry_run()

        self.record_mapping()

        self._copy_backup()
        
        if self.rebalance_downsample:
            self.resample()

    def check_clean_sample_by_rank_distance(self):
        """check if a sample is clean or not (smallest distance in each class)
        """
        print("[TRAIN] Check Pseudo fewshot by pathlist from web domain")
        imgs_by_target = {}
        num_fewshots_by_target = []
        for sample, meta_weight in zip(self.samples, self.meta_weights):
            meta_weight_classes, meta_weight_sims, meta_weight_dists = meta_weight
            for meta_wgt_class, meta_wgt_dist in zip(meta_weight_classes, meta_weight_dists):
                ### 每个样本都要存入所有其所属的类别
                if not (meta_wgt_class in imgs_by_target):
                    imgs_by_target[meta_wgt_class] = []
                ### 其中, meta_weight包括(meta_class, meta_weight, meta_dist)三部分
                imgs_by_target[meta_wgt_class].append([sample, float(meta_wgt_dist)])

        samples_fewshot_pseudo = {}
        target_list = list(imgs_by_target.keys())
        for target in target_list:
            img_list = imgs_by_target[target]
            if self.topk < 0:
                ## random sample images instead of sorted
                random.shuffle(img_list)
                img_list_sorted = img_list
            else:
                ### rerank distance
                img_list_sorted = sorted(img_list, key=lambda x:x[1], reverse=False)
            num_imgs_sorted = max(min(len(img_list), abs(self.topk)), 1)  ## 正样本
            num_imgs_non_sorted = min(max(len(img_list)-num_imgs_sorted, 0), abs(self.topk))  ## 负样本
            num_fewshots_by_target.append([num_imgs_sorted+num_imgs_non_sorted, target])
            img_list_sampled = img_list_sorted[:num_imgs_sorted]
            for img_item in img_list_sampled:
                tfrecord, offset = img_item[0]
                sample_name = str(tfrecord) + "@" + str(offset)
                if not (sample_name in samples_fewshot_pseudo):
                    samples_fewshot_pseudo[sample_name] = [[], []]
                samples_fewshot_pseudo[sample_name][0].append(target)
            if num_imgs_non_sorted > 0:
                img_list_sampled = img_list_sorted[-num_imgs_non_sorted:]
                for img_item in img_list_sampled:
                    tfrecord, offset = img_item[0]
                    sample_name = str(tfrecord) + "@" + str(offset)
                    if not (sample_name in samples_fewshot_pseudo):
                        samples_fewshot_pseudo[sample_name] = [[], []]
                    samples_fewshot_pseudo[sample_name][1].append(target)

        for idx, sample in enumerate(self.samples):
            tfrecord, offset = sample
            sample_name = str(tfrecord) + "@" + str(offset)
            if sample_name in samples_fewshot_pseudo:
                ### 修改这部分样本的domain标签为伪fewshot标签
                self.domains[idx] = samples_fewshot_pseudo[sample_name]

        num_fewshots_by_target = sorted(num_fewshots_by_target, key=lambda x:x[0])
        num_fewshots = [x[0] for x in num_fewshots_by_target]
        print("=========><=========")
        print("number of pseudo few-shot samples {} with {} unique samples".format(
            sum(num_fewshots), len(samples_fewshot_pseudo)
        ))
        print("min fewshots | class", num_fewshots_by_target[0])
        print("max fewshots | class", num_fewshots_by_target[-1])
        print("=========><=========")
        return

    def keep_sample_w_positive_similarity(self):
        ## 使用meta_weights来对所有样本进行筛选 仅仅保留正样本集合
        valid_indexes = []
        for idx, meta_weight in enumerate(self.meta_weights):
            for meta_weight_i in meta_weight[1]:
                if float(meta_weight_i) >= 0:
                    valid_indexes.append(idx)
                    break
        self.samples = [self.samples[index] for index in valid_indexes]
        self.targets = [self.targets[index] for index in valid_indexes]
        self.domains = [self.domains[index] for index in valid_indexes]
        self.meta_weights = [self.meta_weights[index] for index in valid_indexes]
        return

    def load_by_json(self):
        print("[TRAIN] Load samples by pseudo label json for web domain")
        with open(self.annotation, "r") as fr:
            annotation_json = json.load(fr)
        samples = annotation_json['samples']
        root_dirs = annotation_json["roots"]
        index2roots = annotation_json["index2root"]
        samples_full = []
        for sample, root_dir in zip(samples, root_dirs):
            ## 增加根目录
            tf_record, offset = sample
            tf_record_full = os.path.join(index2roots[str(root_dir)], tf_record)
            samples_full.append([tf_record_full, offset])
        self.samples = []
        self.targets = []
        self.domains = []
        self.meta_weights = []
        for sample, target, domain in zip(samples_full, annotation_json['targets'], annotation_json['domains']):
            if len(target) == 0:
                continue
            self.samples.append(sample)
            self.domains.append(domain)
            self.targets.append(target)
            self.meta_weights.append([target, [1.0 for _ in range(len(target))], [0.0 for _ in range(len(target))]])
        print("=========><=========")
        print("[AnnotationJson] #{} samples; #{} labels; #{} domains; #{} meta weights".format(
            len(self.samples), len(self.targets), len(self.domains), len(self.meta_weights)))
        print("=========><=========")
        return

    def load_by_txt(self):
        self.samples = []
        self.targets = []
        self.meta_weights = []
        self.visited_class = set()
        with open(self.pathlist, "r") as fr:
            for line in fr.readlines():
                tf_record_offset = line.strip().split(" ")[0]
                tf_record = tf_record_offset.split("@")[0]
                offset = int(tf_record_offset.split("@")[1])
                tf_record_path = os.path.join(self.root_dir, tf_record)
                meta_info = line.strip().replace(tf_record_offset + " ", "")
                if self.mode != "test":
                    json_info = json.loads(meta_info)
                    ### 网图标签
                    targets = json_info["meta_label"]
                    if len(targets) == 0:
                        continue
                    target_sims = json_info["meta_sim"]
                    target_dists = json_info["meta_dist"]
                    ### 默认拆解成多类sigmoid
                    self.samples.append([tf_record_path, offset])
                    self.targets.append(targets)
                    self.meta_weights.append([targets, target_sims, target_dists])
                    for target in targets:
                        self.visited_class.add(target)

                elif self.mode == "test":
                    targets = meta_info.strip().split(",")
                    targets = [int(target) for target in targets]
                    self.samples.append([tf_record_path, offset])
                    self.targets.append(targets)
                    for target in targets:
                        self.visited_class.add(target)
                    self.meta_weights.append([targets,
                        [1.0 for _ in range(len(targets))], [0.0 for _ in range(len(targets))]])

        if self.mode == "train" or (self.mode == "fewshot" and not self.use_fewshot):
            self.domains = [0 for _ in range(len(self.samples))]
            print("[TRAIN/FewShot] Load samples by pathlist for web domain")
        elif self.mode == "test":
            self.domains = [1 for _ in range(len(self.samples))]
            print("[TEST] Load samples by pathlist for target domain")
        else:
            raise ValueError("mode should only be in train|test|fewshot")
        print("=========><=========")
        print("number of samples", len(self.samples),\
            "number of labels", len(self.targets),\
                "number of domain labels", len(self.domains),\
                    "number of meta weights", len(self.meta_weights),\
                        "number of visited classes", len(self.visited_class))
        print("=========><=========")
        return

    def load_target_by_txt(self):
        ### 如果不存在对应路径则直接返回
        if self.root_dir_target != "" and \
            self.pathlist_target != "" and \
                os.path.exists(self.root_dir_target) and \
                    os.path.exists(self.pathlist_target):
            samples_supp = []
            targets_supp = []
            meta_weights_supp = []
            visited_class_supp = set()
            with open(self.pathlist_target, "r") as fr:
                for line in fr.readlines():
                    tf_record_offset = line.strip().split(" ")[0]
                    meta_info = line.strip().replace(tf_record_offset + " ", "")
                    tf_record = tf_record_offset.split("@")[0]
                    offset = int(tf_record_offset.split("@")[1])
                    tf_record_path = os.path.join(self.root_dir_target, tf_record)
                    # assert(os.path.isfile(tf_record_path))
                    json_info = json.loads(meta_info)
                    target = json_info["label"]
                    meta_weights_supp.append([target,
                        [1.0 for _ in range(len(target))], [0.0 for _ in range(len(target))]])
                    assert (target<self.num_class)
                    samples_supp.append([tf_record_path, offset])
                    targets_supp.append(target)
                    visited_class_supp.add(target)
            domains_supp = [1 for _ in range(len(samples_supp))]

            if self.mode == "train" and self.use_fewshot:
                print("[TRAIN] Load samples by pathlist for target domain")
                if self.fast_eval:
                    ## 使用抽样每类样本快速验证
                    imgs_by_target = {}
                    imgs_by_target_fewshot = {}
                    for sample, target, domain in zip(self.samples, self.targets, self.domains):
                        if not target in imgs_by_target:
                            imgs_by_target[target] = []
                        imgs_by_target[target].append([sample, target, domain])
                    for sample, target, domain in zip(samples_supp, targets_supp, domains_supp):
                        if not target in imgs_by_target_fewshot:
                            imgs_by_target_fewshot[target] = []
                        imgs_by_target_fewshot[target].append([sample, target, domain])
                    print("Number of few-shots", len(imgs_by_target_fewshot))
                    if self.fast_eval == 1:
                        ### webvision
                        target_list = sorted(list(imgs_by_target.keys()))[:50]
                        target_must_include = [542, 428, 134, 517, 677, 998]
                        target_list += target_must_include
                    elif self.fast_eval == 2:
                        ### google500
                        target_list = sorted(list(imgs_by_target.keys()))[:50]
                        target_must_include = [137, 246, 385, 227]
                        target_list += target_must_include
                    else:
                        raise ValueError("fast eval should be in 1|2")
                    print("Fast Eval of WebVision Train Class ID", target_list)
                    samples_fewshot = []
                    targets_fewshot = []
                    domains_fewshot = []
                    meta_weights_fewshot = []
                    for target in target_list:
                        img_list = imgs_by_target[target]
                        img_list_fewshot = imgs_by_target_fewshot[target]
                        if target in target_must_include:
                            ### 使用所有网图训练样本数据
                            img_list_sampled = img_list
                        else:
                            ### 仅抽样部分网图训练样本数据
                            img_list_sampled = random.sample(img_list, min(200, len(img_list)))
                        ### 加入FewShot样本数据
                        img_list_sampled += img_list_fewshot
                        for img_item in img_list_sampled:
                            samples_fewshot.append(img_item[0])
                            targets_fewshot.append(img_item[1])
                            domains_fewshot.append(img_item[2])
                            meta_weights_fewshot.append([img_item[1],
                                [1.0 for _ in range(len(img_item[1]))],
                                    [0.0 for _ in range(len(img_item[1]))]])
                    self.samples = samples_fewshot
                    self.targets = targets_fewshot
                    self.domains = domains_fewshot
                    self.meta_weights = meta_weights_fewshot
                else:
                    ## 正常训练web domain + target domain FewShot
                    self.samples += samples_supp
                    self.targets += targets_supp
                    self.domains += domains_supp
                    self.meta_weights += meta_weights_supp
    
            elif self.mode == "fewshot" and self.use_fewshot:
                print("[FEW-SHOT] Load samples by pathlist for target domain")
                ## 使用真实target domain的fewshot样本
                self.samples = samples_supp
                self.targets = targets_supp
                self.domains = domains_supp
                self.meta_weights = meta_weights_supp

            print("=========><=========")
            print("number of samples", len(self.samples),\
                    "number of labels", len(self.targets),\
                        "number of domains", len(self.domains),\
                            "number of meta-weights", len(self.meta_weights),\
                                "number of classes", len(visited_class_supp))
            print("=========><=========")
        
        else:
            if self.mode == "fewshot" and (not self.use_fewshot):
                print("[FEW-SHOT] Load samples by pathlist from web domain")
                ## 使用抽样的web domain的样本假装fewshot样本
                imgs_by_target = {}
                num_fewshots_by_target = []
                for sample, meta_weight in zip(self.samples, self.meta_weights):
                    meta_weight_classes, meta_weight_sims, meta_weight_dists = meta_weight
                    for meta_wgt_class, meta_wgt_dist in zip(meta_weight_classes, meta_weight_dists):
                        ### 每个样本都要存入所有其所属的类别
                        if not (meta_wgt_class in imgs_by_target):
                            imgs_by_target[meta_wgt_class] = []
                        ### 其中, meta_weight包括(meta_class, meta_weight, meta_dist)三部分
                        imgs_by_target[meta_wgt_class].append([sample, float(meta_wgt_dist)])

                samples_fewshot_pseudo = {}
                target_list = list(imgs_by_target.keys())
                for target in target_list:
                    img_list = imgs_by_target[target]
                    if self.topk < 0:
                        ## random sample images instead of sorted
                        random.shuffle(img_list)
                        img_list_sorted = img_list
                    else:
                        ### rerank distance
                        img_list_sorted = sorted(img_list, key=lambda x:x[1], reverse=False)
                    num_imgs_sorted = max(min(len(img_list), abs(self.topk)), 1)  ## 正样本
                    num_imgs_non_sorted = min(max(len(img_list)-num_imgs_sorted, 0), abs(self.topk))  ## 负样本
                    num_fewshots_by_target.append([num_imgs_sorted+num_imgs_non_sorted, target])
                    img_list_sampled = img_list_sorted[:num_imgs_sorted]
                    for img_item in img_list_sampled:
                        tfrecord, offset = img_item[0]
                        sample_name = str(tfrecord) + "@" + str(offset)
                        if not (sample_name in samples_fewshot_pseudo):
                            samples_fewshot_pseudo[sample_name] = [[], []]
                        samples_fewshot_pseudo[sample_name][0].append(target)
                    if num_imgs_non_sorted > 0:
                        img_list_sampled = img_list_sorted[-num_imgs_non_sorted:]
                        for img_item in img_list_sampled:
                            tfrecord, offset = img_item[0]
                            sample_name = str(tfrecord) + "@" + str(offset)
                            if not (sample_name in samples_fewshot_pseudo):
                                samples_fewshot_pseudo[sample_name] = [[], []]
                            samples_fewshot_pseudo[sample_name][1].append(target)

                samples_fewshot = []
                targets_fewshot = []
                domains_fewshot = []
                meta_weights_fewshot = []
                for idx, sample in enumerate(self.samples):
                    tfrecord, offset = sample
                    sample_name = str(tfrecord) + "@" + str(offset)
                    if sample_name in samples_fewshot_pseudo:
                        ### 修改这部分样本的domain标签为伪fewshot标签
                        samples_fewshot.append(self.samples[idx])
                        targets_fewshot.append(self.targets[idx])
                        domains_fewshot.append(samples_fewshot_pseudo[sample_name])
                        meta_weights_fewshot.append(self.meta_weights[idx])   

                num_fewshots_by_target = sorted(num_fewshots_by_target, key=lambda x:x[0])
                num_fewshots = [x[0] for x in num_fewshots_by_target]

                self.samples = samples_fewshot
                self.targets = targets_fewshot
                self.domains = domains_fewshot
                self.meta_weights = meta_weights_fewshot
                
                print("=========><=========")
                print("number of samples", len(self.samples),\
                        "number of labels", len(self.targets),\
                            "number of domains", len(self.domains),\
                                "number of meta-weights", len(self.meta_weights),\
                                    "number of classes", len(num_fewshots_by_target))
                print("number of pseudo few-shot samples {} with {} unique samples".format(
                    sum(num_fewshots), len(samples_fewshot_pseudo)
                ))
                print("min fewshots | class", num_fewshots_by_target[0])
                print("max fewshots | class", num_fewshots_by_target[-1])
                print("=========><=========")
            
            else:
                print("=========><=========")
                if self.root_dir_target != "" or self.pathlist_target != "":
                    print("Invalid Pathlist for FewShot Target")
                    print(self.root_dir_target)
                    print(self.pathlist_target)
                else:
                    print("Empty Pathlist for FewShot Target")
                print("=========><=========")
        return

    def get_default_sample_weight(self):
        print("==============>Use Default Weights Reweighting<==============")
        class_nums = [1.0 for _ in range(self.num_class)]
        for targets in self.targets:
            for target in targets:
                class_nums[int(target)] += 1
        class_nums = np.array(class_nums)
        class_median = np.median(class_nums)
        class_weights = class_nums / class_median
        self.class_weights = class_weights
        self.pos_weights = np.sqrt((len(self.samples) - class_nums) / class_nums)
        return

    def get_scaled_sample_weight(self):
        print("==============>Use Meta Weights Reweighting<==============")
        ### 重新计算每个类别的样本数目并且求反比==>top 1
        class_nums = [1 for _ in range(self.num_class)]
        class_sims_sum = [1 for _ in range(self.num_class)]

        ### 统计每个数量并得到中位数
        for targets, meta_weights in zip(self.targets, self.meta_weights):
            for target, meta_weight in zip(targets, meta_weights[1]):
                class_nums[int(target)] += 1
                class_sims_sum[int(target)] += meta_weight
        ### 仅仅计算第一个类别的交叉熵损失=>使用第一个类别权重
        class_nums, class_sims_sum = np.array(class_nums), np.array(class_sims_sum)
        self.pos_weights = np.sqrt((len(self.samples) - class_nums) / class_nums)
        class_median = np.median(class_nums)
        class_weights = class_nums / class_median
        class_scales = class_nums / class_sims_sum
        class_weights *= class_scales
        ### 二选一
        self.class_weights = class_weights
        return

    def record_mapping(self):
        print("Preparing mapping from image path to image index number")
        self.samples_index = []
        save_mapping_index = os.path.join(self.save_dir, "index_mapping_{}.txt".format(self.mode))
        with open(save_mapping_index, "w") as f_write:
            for index, item in enumerate(self.samples):
                tfrecord, offset = item
                target = self.targets[index]
                path = tfrecord + "@" + str(offset)
                self.samples_index.append(index)
                f_write.write(" ".join([str(index), str(path), str(target)]) + "\n")
        return

    def _copy_backup(self):
        self.samples_copy = copy(self.samples)
        self.targets_copy = copy(self.targets)
        self.domains_copy = copy(self.domains)
        self.meta_weights_copy = copy(self.meta_weights)
        self.samples_index_copy = copy(self.samples_index)
        return

    def _dry_run(self):
        if self.mode == "train":
            target_ids = set()
            random_index = []
            for idx, targets in enumerate(self.targets):
                is_new_added = False
                for target_id in targets:
                    if not (target_id in target_ids):
                        is_new_added = True
                        break
                if is_new_added:
                    for target_id in targets:
                        target_ids.add(target_id)
                    random_index.append(idx)
            random_index += np.arange(len(self.samples)-32, len(self.samples)).tolist()
            random.shuffle(random_index)
            self.samples = [self.samples[idx] for idx in random_index]
            self.targets = [self.targets[idx] for idx in random_index]
            self.domains = [self.domains[idx] for idx in random_index]
            self.meta_weights = [self.meta_weights[idx] for idx in random_index]
        else:
            ####保证每个类都有一个样本进行调试
            samples_dry_run, targets_dry_run,\
                domains_dry_run, meta_weights_dry_run = [], [], [], []
            visited_class_dry_run = set()
            for sample, target, domain, meta_weight in zip(self.samples,
                self.targets, self.domains, self.meta_weights):
                for target_id in target:
                    if not (target_id in visited_class_dry_run):
                        flag_add_sample = True
                        break
                    else:
                        flag_add_sample = False
                
                if flag_add_sample:
                    for target_id in target:
                        visited_class_dry_run.add(target_id)
                    samples_dry_run.append(sample)
                    targets_dry_run.append(target)
                    domains_dry_run.append(domain)
                    meta_weights_dry_run.append(meta_weight)

            self.samples = samples_dry_run
            self.targets = targets_dry_run
            self.domains = domains_dry_run
            self.meta_weights = meta_weights_dry_run

        print("=========><=========")
        print("[Dry-run] #{} samples; #{} labels; #{} domains; #{} meta weights".format(
            len(self.samples), len(self.targets), len(self.domains), len(self.meta_weights)))
        print("=========><=========")
        return

    def _parser(self, feature_list):
        """get the image file and perform transformation
        feature_list: the dictionary to load features (images)
        """
        for key, feature in feature_list: 
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = Image.open(io.BytesIO(image_raw))
                image = image.convert('RGB')
                return image
        return

    def get_tfrecord_image(self, record_file, offset):
        """read images from tfrecord"""
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
        image = self._parser(feature)
        return image

    def resample(self):
        print('==>down-sampling dataset scale for balancing<==')
        assert(self.mode == "train"), "repeat for data-resampling is only valid for traininig mode"
        ## 需要更新4个列表 samples; targets; domains; sample_index
        if self.imgs_by_class == None:
            imgs_by_class = {}
            ## 先按照类别整理所有图像样本
            for im, lab, domain, meta_weight, im_index in zip(self.samples_copy, self.targets_copy,\
                self.domains_copy, self.meta_weights, self.samples_index_copy):
                if not (lab in imgs_by_class):
                    imgs_by_class[lab] = [[], []]   # 第一个列表存储fewshot样本 第二个列表存储非fewshot样本
                if domain > 0:
                    imgs_by_class[lab][0].append([im, lab, domain, meta_weight, im_index])
                else:
                    imgs_by_class[lab][1].append([im, lab, domain, meta_weight, im_index])
            self.imgs_by_class = imgs_by_class
        
        samples_balance = []
        targets_balance = []
        domains_balance = []
        meta_weights_balance = []
        samples_index_balance = []
        N_ratio = 2
        for lab in self.imgs_by_class.keys():
            ## 从平衡的样本集中随机抽取固定数目的样本
            imgs_all_lab = self.imgs_by_class[lab]
            imgs_fewshot_lab = imgs_all_lab[0]
            num_fewshot_imgs = len(imgs_fewshot_lab)
            imgs_web_lab = imgs_all_lab[1]
            num_web_imgs = len(imgs_web_lab)
            num_web_imgs_sampled = min(max(num_web_imgs//N_ratio, 1), num_web_imgs)
            imgs_web_sampled_lab = random.sample(imgs_web_lab, num_web_imgs_sampled)
            # print("category {} has {} web images and {} fewshot images".format(lab,\
            #     num_web_imgs_sampled, num_fewshot_imgs))
            imgs_sampled_lab = imgs_web_sampled_lab + imgs_fewshot_lab
            for item in imgs_sampled_lab:
                samples_balance.append(item[0])
                targets_balance.append(item[1])
                domains_balance.append(item[2])
                meta_weights_balance.append(item[3])
                samples_index_balance.append(item[4])
        ## 更新样本集
        self.samples = samples_balance
        self.targets = targets_balance
        self.domains = domains_balance
        self.meta_weights = meta_weights_balance
        self.samples_index = samples_index_balance
        print('=> done {} / {} number of resampled items'.format(len(self.samples), self.__len__()))
        
        return

    def repeat(self):
        print('=> repeating dataset for balancing')
        assert(self.mode == "train"), "repeat for data-resampling is only valid for traininig mode"
        ## 需要更新4个列表 samples; targets; domains; sample_index
        targets = np.array(self.targets_copy)
        uniq, freq = np.unique(targets, return_counts=True)
        inv = (1/freq)**0.5
        p = inv/inv.sum()
        weight = (10 * p)/(p.min())
        weight = weight.astype(int)
        weight = {u:w for u,w in zip(uniq, weight)}
        ## 从初始化的所有数据列表中进行re-balance抽样
        samples_balance = []
        targets_balance = []
        domains_balance = []
        meta_weights_balance = []
        samples_index_balance = []
        for im, lab, domain, meta_weight, im_index in zip(self.samples_copy, self.targets_copy,\
            self.domains_copy, self.meta_weights_copy, self.samples_index_copy):
            samples_balance += [im]*weight[lab]
            targets_balance += [lab]*weight[lab]
            domains_balance += [domain]*weight[lab]
            meta_weights_balance += [meta_weight]*weight[lab]
            samples_index_balance += [im_index]*weight[lab]
        ## 更新样本集
        self.samples = []
        self.targets = []
        self.domains = []
        self.meta_weights = []
        self.samples_index = []
        ## 从平衡的样本集中随机抽取固定数目的样本
        index_shuf = list(range(len(samples_balance)))
        random.shuffle(index_shuf)
        for i in index_shuf[:len(self.targets_copy)]:
            self.samples.append(samples_balance[i])
            self.targets.append(targets_balance[i])
            self.domains.append(domains_balance[i])
            self.meta_weights.append(meta_weights_balance[i])
            self.samples_index.append(samples_index_balance[i])
        print('=> done')
        return

    def set_alb_transform(self):
        self.albs_transform_color = [alb.Equalize(p=0.5), alb.ColorJitter(p=0.5),\
            alb.ToGray(p=0.5), alb.Sharpen(p=0.5), alb.HueSaturationValue(p=0.5),\
                alb.RandomBrightness(p=0.5), alb.RandomBrightnessContrast(p=0.5),\
                    alb.RandomToneCurve(p=0.5)]
        self.albs_transform_basic = [RandomBorder(), RandomTranslate(),\
            RandomTextOverlay(), RandomStripesOverlay(),
                alb.OpticalDistortion(p=0.5),\
                    alb.GridDistortion(p=0.5, border_mode=cv2.BORDER_REPLICATE)]
        self.albs_transform_noise = [alb.ISONoise(p=0.5), alb.RandomFog(p=0.5, fog_coef_upper=0.5),\
            alb.RandomSnow(p=0.5, brightness_coeff=1.2), alb.RandomRain(p=0.5, drop_length=5),\
                alb.RandomShadow(p=0.5, num_shadows_lower=0, num_shadows_upper=1),\
                    alb.GaussNoise(p=0.5), alb.ImageCompression(quality_lower=95, p=1),\
                        alb.MotionBlur(p=0.5), alb.Blur(p=1),\
                            alb.GaussianBlur(p=0.5), alb.GlassBlur(sigma=0.2, p=1)]
        return

    def alb_transform(self, image):
        """use albumentation transform for data augmentation"""
        image = np.array(image)
        alb_trans = [random.choice(self.albs_transform_basic),\
            random.choice(self.albs_transform_noise)]
        if not self.no_color_transform:
            alb_trans.append(random.choice(self.albs_transform_color))
        alb_trans = alb.Compose(random.sample(alb_trans, random.randint(1, len(alb_trans))))
        image_aug = alb_trans(image=image)['image']
        image_aug = Image.fromarray(image_aug)
        return image_aug

    def __getitem__(self, index):
        index = index % self.__len__()
        tfrecord, offset = self.samples[index]
        path = tfrecord + "@" + str(offset)
        image = self.get_tfrecord_image(tfrecord, offset)
        img = self.transform(image)
        targets = self.targets[index]
        target_meta_soft = None
        domains = self.domains[index]
        domain = torch.zeros(self.num_class)
        if domains == 0:
            ## 该样本任意类不是fewshot样本
            pass
        elif domains == 1:
            ## 该样本所有target类均为fewshot样本
            for target in targets:
                domain[int(target)] = 1.
        elif type(domains) is list:
            ## 该样本根据rank排序设置正样本/负样本
            for target in domains[0]:
                domain[int(target)] = 1.
            for target in domains[1]:
                domain[int(target)] = -1

        if self.mode == "train":
            meta_label, meta_sim, meta_dist = self.meta_weights[index]
            if self.use_meta_weights > 0:
                ### sample weight可以由meta置信度决定
                sample_weight = max(meta_sim)
            elif self.use_meta_weights == 0:
                ### sample weight仅仅由各类别数目决定
                sample_weight = max([self.class_weights[target] for target in targets])
            elif self.use_meta_weights == -1:
                ### sample weight直接返回的是1.0
                sample_weight = 1.0
            
            ### 基于similarity构造probability(概率和为1)
            meta_label, meta_sim = np.array(meta_label), np.array(meta_sim)
            meta_sim_prob = np.exp(meta_sim/0.1)/sum(np.exp(meta_sim/0.1))
            if self.use_meta_weights == 2:
                ### 使用meta信息 Multi-label抽样单标签
                targets = np.random.choice(meta_label, p=meta_sim_prob)
            elif self.use_meta_weights == 3:
                ### 使用meta信息 Multi-label软标签
                target_meta_soft = torch.zeros(self.num_class)
                target_meta_soft[torch.Tensor(meta_label).long()] = torch.Tensor(meta_sim_prob)

            if self.no_color_transform:
                rotate_deg = np.random.randint(low=-30, high=30)
                image = rotate_and_crop(image, rotate_deg)
            
            # img_aug = self.transform_strong(self.alb_transform(image))
            img_aug = self.transform_strong(image)

            target_multi_hot = torch.zeros((self.num_class,))
            for target in set(targets):
                target_multi_hot[int(target)] = 1
            if target_meta_soft is None:
                target_meta_soft = target_multi_hot
            
            return img, target_multi_hot, img_aug, domain,\
                    path, torch.Tensor([self.samples_index[index]]),\
                        torch.Tensor([sample_weight]), target_meta_soft
        
        elif self.mode == "fewshot" or self.mode == "test":
            target_multi_hot = torch.zeros((self.num_class,))
            for target in set(targets):
                target_multi_hot[int(target)] = 1

            return img, target_multi_hot, domain,\
                path, torch.Tensor([self.samples_index[index]])
           
    def __len__(self):
        return len(self.samples)


class nuswide_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, pathlist,\
        root_dir_test_web, pathlist_test_web, root_dir_test_target, pathlist_test_target,\
            distributed, crop_size=0.5, root_dir_target="", pathlist_target="",\
                save_dir="", dry_run=False, use_fewshot=True, annotation="",\
                    no_color_transform=False, fast_eval=False, eval_only=False,\
                        rebalance_downsample=False, use_meta_weights=0,\
                            drop_last=True, topk=50):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.use_fewshot = use_fewshot
        self.use_meta_weights = use_meta_weights
        self.pathlist = pathlist
        self.annotation = annotation
        self.distributed = distributed
        self.fast_eval = fast_eval
        self.root_dir_test_web = root_dir_test_web
        self.pathlist_test_web = pathlist_test_web
        self.root_dir_test_target = root_dir_test_target
        self.pathlist_test_target = pathlist_test_target
        self.root_dir_target = root_dir_target
        self.pathlist_target = pathlist_target
        self.rebalance_downsample = rebalance_downsample
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.drop_last = drop_last
        self.topk = topk
        ### 减少CROP_SIZE的scale会进一步增加数据增强尺度
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            FancyPCA(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.no_color_transform = no_color_transform
        if no_color_transform:
            self.transform_strong = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                FancyPCA(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_strong = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),            
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                FancyPCA(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                FancyPCA(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.eval_only = eval_only
        if eval_only:
            self.transform_train = self.transform_test
            self.transform_strong = self.transform_test

    def run(self):
        print("------------------------------------------------------")
        print("TRAINING SET WEB")
        save_dir_train = os.path.join(self.save_dir, "data_train")
        os.makedirs(save_dir_train, exist_ok=True)
        train_dataset = nuswide_dataset(root_dir=self.root_dir, pathlist=self.pathlist, transform=self.transform_train,\
            mode="train", num_class=self.num_class, transform_strong = self.transform_strong,\
                root_dir_target=self.root_dir_target, pathlist_target=self.pathlist_target,\
                    save_dir=save_dir_train, dry_run=self.dry_run,\
                        use_fewshot=self.use_fewshot, annotation=self.annotation,\
                            no_color_transform=self.no_color_transform,\
                                fast_eval=self.fast_eval,\
                                    rebalance_downsample=self.rebalance_downsample,\
                                        use_meta_weights=self.use_meta_weights,\
                                            topk=self.topk)
        print("------------------------------------------------------")
        print("FEW-SHOT SET TARGET")
        save_dir_fewshot = os.path.join(self.save_dir, "data_fewshot")
        os.makedirs(save_dir_fewshot, exist_ok=True)
        fewshot_dataset = nuswide_dataset(root_dir=self.root_dir, pathlist=self.pathlist, transform=self.transform_train,\
            mode="fewshot", num_class=self.num_class, transform_strong = self.transform_strong,\
                root_dir_target=self.root_dir_target, pathlist_target=self.pathlist_target,\
                    save_dir=save_dir_fewshot, dry_run=self.dry_run,\
                        use_fewshot=self.use_fewshot, topk=self.topk)
        print("------------------------------------------------------")
        print("TESTING SET WEB")
        save_dir_test_web = os.path.join(self.save_dir, "data_test_web")
        os.makedirs(save_dir_test_web, exist_ok=True)
        test_dataset_web = nuswide_dataset(root_dir=self.root_dir_test_web, pathlist=self.pathlist_test_web,\
            transform=self.transform_test, mode='test', num_class=self.num_class,\
                save_dir=save_dir_test_web, dry_run=self.dry_run)
        print("------------------------------------------------------")
        print("TESTING SET TARGET")
        save_dir_test_imgnet = os.path.join(self.save_dir, "data_test_imgnet")
        os.makedirs(save_dir_test_imgnet, exist_ok=True)
        test_dataset_target = nuswide_dataset(root_dir=self.root_dir_test_target, pathlist=self.pathlist_test_target,\
            transform=self.transform_test, mode='test', num_class=self.num_class,\
                save_dir=save_dir_test_imgnet, dry_run=self.dry_run)
        print("------------------------------------------------------")
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            fewshot_sampler = torch.utils.data.distributed.DistributedSampler(fewshot_dataset, shuffle=False)
            test_sampler_web = torch.utils.data.distributed.DistributedSampler(test_dataset_web, shuffle=False)
            test_sampler_target = torch.utils.data.distributed.DistributedSampler(test_dataset_target, shuffle=False)
        else:
            self.train_sampler = None
            fewshot_sampler = None
            test_sampler_web = None
            test_sampler_target = None

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=self.drop_last)                                              

        fewshot_loader = DataLoader(
            dataset=fewshot_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=fewshot_sampler)   
             
        test_loader_web = DataLoader(
            dataset=test_dataset_web,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=test_sampler_web)                             

        test_loader_target = DataLoader(
            dataset=test_dataset_target,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=test_sampler_target)
    
        return train_loader, fewshot_loader, test_loader_web, test_loader_target

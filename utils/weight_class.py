import numpy as np
import json


def extract_class_weight(pathlist, N_class):
    labels = [1. for _ in range(N_class)]
    with open(pathlist, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 00052-of-00064.tfrecord@101027009 {"conf_score": 1.0, "label": 0, "sample_weight": 1.2033292383292387}
            img_path = line.strip().split(" {")[0]
            json_path = line.strip().replace(img_path, "")
            label_json = json.loads(json_path.strip())
            label_idx = int(label_json["label"])
            label_weight = float(label_json["sample_weight"])
            labels[label_idx] = label_weight
    return labels


def get_default_sample_weight(pathlist, N_class):
    print("==============>Use Default Weights Reweighting<==============")
    num_samples_all = 0
    class_nums = [1.0 for _ in range(N_class)]
    with open(pathlist, "r") as fr:
        for line in fr.readlines():
            img_path = line.strip().split(" ")[0]
            num_samples_all += 1
            json_str = line.strip().replace(img_path+" ", "")
            info = json.loads(json_str)
            targets = info["label"]
            for target in targets:
                class_nums[int(target)] += 1

    class_nums = np.array(class_nums)
    class_median = np.median(class_nums)
    class_weights = class_nums / class_median
    pos_ratio = class_nums / num_samples_all
    neg_ratio = (num_samples_all - class_nums) / num_samples_all
    pos_weights = (num_samples_all - class_nums) / class_nums
    print("positive ratio", pos_ratio)
    print("minimum", np.amin(pos_ratio), "maximum", np.amax(pos_ratio))
    print("negative ratio", neg_ratio)
    print("minimum", np.amin(neg_ratio), "maximum", np.amax(neg_ratio))
    print("positive ratio median", np.median(pos_ratio), "negative ratio median", np.median(neg_ratio))
    print("positive ratio avg.", np.mean(pos_ratio), "negative ratio avg.", np.mean(neg_ratio))
    print("positive weight", pos_weights)
    print("all pos/(pos+neg)=", np.sum(class_nums)/(num_samples_all*N_class))
    return


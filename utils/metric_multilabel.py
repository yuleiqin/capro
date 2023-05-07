import numpy as np
from scipy.stats.mstats import gmean
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
import os
import torch
import torch.nn as nn
import json


def sample_class_similarity(args):
    ## 计算每个类别之间的相似度并进行采样
    if args.class_sim_path != "" and os.path.exists(args.class_sim_path):
        with open(args.class_sim_path, "r") as fr:
            class_sim_raw = json.load(fr)
        class_sim = []
        for class_idx in range(args.num_class):
            ## 计算class similarity
            class_sim_raw_idx = class_sim_raw[str(class_idx)]
            class_sim_pos_ids = [int(class_idx)]
            class_sim_pos_sims = [1.0]
            class_sim_neg_ids = []
            for class_sim_pos_id, class_sim_pos_sim in class_sim_raw_idx:
                class_sim_pos_ids.append(int(class_sim_pos_id))
                class_sim_pos_sims.append(float(class_sim_pos_sim))
            class_sim_pos_ids, class_sim_pos_sims = np.array(class_sim_pos_ids), np.array(class_sim_pos_sims)
            class_sim_pos_sims /= np.sum(class_sim_pos_sims)
            for class_neg_id in range(args.num_class):
                if not (class_neg_id in class_sim_pos_ids):
                    class_sim_neg_ids.append(class_neg_id)
            class_sim_neg_ids = np.array(class_sim_neg_ids)
            class_sim.append([class_sim_pos_ids, class_sim_pos_sims, class_sim_neg_ids])
        criterion_triplet = nn.TripletMarginWithDistanceLoss(distance_function=\
            lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.1, reduction='none').cuda(args.gpu)
    else:
        class_sim = None
        criterion_triplet = None
    return class_sim, criterion_triplet


def sigmoid_loss_with_posweight(prediction, target, criterion_pcl, pos_weight, sample_weight, target_valid):
    """
    prediction: sigmoid-activated logits
    target: multi-hot vector logtis
    """
    prediction_squeeze = prediction.view(-1)
    target_squeeze = target.view(-1)
    pos_weight_target = torch.ones_like(target).view(-1)
    pos_weight_repeat = pos_weight.repeat(target.size(0))
    pos_weight_target[target_squeeze>0] = pos_weight_repeat[target_squeeze>0]
    sample_weight_squeeze = sample_weight.repeat_interleave(target.size(1)).view(-1)
    loss_bce_posweight = (criterion_pcl(prediction_squeeze,\
        target_squeeze)*pos_weight_target*target_valid.view(-1)*sample_weight_squeeze).mean()
    return loss_bce_posweight


def precision_recall_map_sklearn(predictions, predictions_prob_all, targets, num_class):
    """Computes the precision/recall/f1/mAP over the k top predictions for the specified values of k
    --predictions: array each row has topK predicted class IDs
    --predictions_prob_all: array each row has all predictions probability
    --targets: array each row has GT class IDs
    --num_class: total number of classes
    """
    y_true = []
    y_pred_bin = []
    for prediction, target in zip(predictions, targets):
        y_pred_bin_i = np.zeros(num_class)
        y_pred_bin_i[prediction] = 1

        y_true_i = np.zeros(num_class)
        target = np.nonzero(target)[0]
        y_true_i[target] = 1

        y_pred_bin.append(y_pred_bin_i)
        y_true.append(y_true_i)
    
    y_pred_bin = np.array(y_pred_bin)
    y_true = np.array(y_true)
    mAP_micro = average_precision_score(y_true, predictions_prob_all, average='micro') * 100
    # mAP_macro = average_precision_score(y_true, predictions_prob_all, average='macro') * 100

    c_precision_sk = precision_score(y_true, y_pred_bin, labels=np.arange(num_class), average='macro') * 100
    c_recall_sk = recall_score(y_true, y_pred_bin, labels=np.arange(num_class), average='macro') * 100
    c_f1_sk = 2 * (c_precision_sk * c_recall_sk) / (c_precision_sk + c_recall_sk)

    o_precision_sk = precision_score(y_true, y_pred_bin, labels=np.arange(num_class), average='micro') * 100
    o_recall_sk = recall_score(y_true, y_pred_bin, labels=np.arange(num_class), average='micro') * 100
    o_f1_sk = 2 * (o_precision_sk * o_recall_sk) / (o_precision_sk + o_recall_sk)

    return c_precision_sk, c_recall_sk, c_f1_sk, o_precision_sk, o_recall_sk, o_f1_sk, mAP_micro    


def precision_recall_map(predictions, predictions_prob_all, targets, num_class):
    """Computes the precision/recall/f1/mAP over the k top predictions for the specified values of k
    --predictions: array each row has topK predicted class IDs
    --predictions_prob_all: array each row has all predictions
    --targets: array each row has GT class IDs
    --num_class: total number of classes
    """
    ### 仅抽取top k个概率最高的预测类别
    ### 注意为了避免计算错误应该使用所有的样本
    ### precision: 预测正确的样本标签数目/预测的样本标签数目
    ### recall: 预测正确的样本标签数目/所有GT样本数目
    ### C-P; C-R: 以每个类为对象统计precision/recall再取平均
    ### O-P; O-R: 以每个样本为对象统计precision/recall再取平均
    c_correct = [0 for _ in range(num_class)]
    c_predict = [1e-4 for _ in range(num_class)]
    c_gt = [1e-4 for _ in range(num_class)]        
    o_correct = []
    o_predict = []
    o_gt = []

    y_true = []
    for prediction, target in zip(predictions, targets):
        y_true_i = np.zeros(num_class)
        target = np.nonzero(target)[0]
        y_true_i[target] = 1
        y_true.append(y_true_i)
        num_correct = 0
        for pred_i in prediction:
            if (pred_i in target):
                ### 当前预测正确的类别++
                c_correct[int(pred_i)] += 1
                num_correct += 1

            ### 所有预测类别++
            c_predict[int(pred_i)] += 1
        
        for target_i in target:
            ### 所有GT类别++
            c_gt[int(target_i)] += 1
        
        o_correct.append(num_correct)
        o_predict.append(len(prediction)+1e-4)
        o_gt.append(len(target)+1e-4)
    
    y_true = np.array(y_true)
    mAP_micro = average_precision_score(y_true, predictions_prob_all, average='micro') * 100
    # mAP_macro = average_precision_score(y_true, predictions_prob_all, average='macro') * 100

    c_correct = np.array(c_correct)
    c_predict = np.array(c_predict)
    c_gt = np.array(c_gt)
    o_correct = np.array(o_correct)
    o_predict = np.array(o_predict)
    o_gt = np.array(o_gt)

    c_precision = float(np.mean(c_correct/c_predict)) * 100.
    c_recall = float(np.mean(c_correct/c_gt)) * 100.
    c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall)
    o_precision = float(np.mean(o_correct/o_predict)) * 100.
    o_recall = float(np.mean(o_correct/o_gt)) * 100.
    o_f1 = 2 * (o_precision * o_recall) / (o_precision + o_recall)

    return c_precision, c_recall, c_f1, o_precision, o_recall, o_f1, mAP_micro

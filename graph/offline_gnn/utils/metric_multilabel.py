import numpy as np
from scipy.stats.mstats import gmean
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
import os


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



if __name__ == "__main__":
    num_class = 81
    # filelist_path = "/youtu-reid/yuleiqin/code_utils/webFG-fewshot-research-v2/dataset/filelist/val_nus_81_tf.txt"

    # print("processing data")
    # predictions = []
    # predictions_prob = []
    # predictions_gt = []
    # predictions_gt_prob = []
    # targets = []

    # with open(filelist_path, "r") as fr:
    #     for line in fr.readlines():
    #         ### gt labels
    #         gt_labels = line.strip().split(" ")[1].split(",")
    #         gt_labels = np.array([int(gt_label) for gt_label in gt_labels])
    #         target = np.zeros(num_class)
    #         target[gt_labels] = 1
    #         targets.append(target)
    #         ### predicted probability
    #         pred_probs = np.random.random(num_class)
    #         pred_labels_sort = np.argsort(pred_probs)[::-1]
    #         pred_labels = pred_labels_sort[:3]
    #         predictions_prob.append(pred_probs[pred_labels])
    #         predictions.append(pred_labels)
    #         ### predicted ground-truth
    #         prediction_gt_labels = gt_labels[:3]
    #         predictions_gt.append(prediction_gt_labels)
    #         predictions_gt_prob.append(np.ones(prediction_gt_labels.shape))

    # predictions_prob = np.array(predictions_prob)
    # predictions = np.array(predictions)
    # predictions_gt = np.array(predictions_gt)
    # predictions_gt_prob = np.array(predictions_gt_prob)
    # targets = np.array(targets)

    root_dir = "/youtu-public/YT_SN5/yolay/proposed_ckpt/web-nuswide-vanilla-pretrained-wgt/stage1"
    pred_npy_path = os.path.join(root_dir, "pred.npy")
    pred_prob_npy_path = os.path.join(root_dir, "pred_prob_all.npy")
    target_npy_path = os.path.join(root_dir, "gt.npy")

    predictions = np.load(pred_npy_path)
    predictions_prob = np.load(pred_prob_npy_path)
    targets = np.load(target_npy_path)

    print("evaluation prediction")
    c_precision, c_recall, c_f1,\
        o_precision, o_recall, o_f1, mAP = precision_recall_map(predictions, predictions_prob,\
            targets, num_class)

    print("C-P={}, C-R={}, C-F1={}, O-P={}, O-R={}, O-F1={}, mAP={}".format(
        c_precision, c_recall, c_f1,\
        o_precision, o_recall, o_f1,\
        mAP,
    ))

    # print("evaluation GT-sample")
    # c_precision, c_recall, c_f1,\
    #     o_precision, o_recall, o_f1, mAP = precision_recall_map(predictions_gt, predictions_gt_prob,\
    #         targets, num_class)

    # print("C-P={}, C-R={}, C-F1={}, O-P={}, O-R={}, O-F1={}, mAP={}".format(
    #     c_precision, c_recall, c_f1,\
    #     o_precision, o_recall, o_f1,\
    #     mAP,
    # ))

import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('seaborn-whitegrid')


save_dir = "f1_figs/results"
os.makedirs(save_dir, exist_ok=True)
save_npy_path = os.path.join(save_dir, "results.npy")
th_score_list = np.linspace(0.01, 0.99, 100)

if not os.path.exists(save_npy_path):
    mapping_idx2wdnet_g500_path = "dataset/filelist/mapping_google_500.txt"
    mapping_idx2wdnet_web1k_path = "dataset/filelist/mapping_webvision_1k.txt"
    mapping_idx2wdnet_g500 = {}
    mapping_wdnet2idx_g500 = {}
    wdnet_g500 = set()
    with open(mapping_idx2wdnet_g500_path, "r") as fr:
        for line in fr.readlines():
            class_idx, wdnet_id = line.strip().split(" ")
            mapping_idx2wdnet_g500[int(class_idx)] = wdnet_id
            mapping_wdnet2idx_g500[wdnet_id] = int(class_idx)
            wdnet_g500.add(wdnet_id)

    wdnet_w1k = set()
    mapping_idx2wdnet_web1k = {}
    with open(mapping_idx2wdnet_web1k_path, "r") as fr:
        for line in fr.readlines():
            class_idx, wdnet_id = line.strip().split(" ")
            mapping_idx2wdnet_web1k[int(class_idx)] = wdnet_id
            wdnet_w1k.add(wdnet_id)

    ### 增加一类
    num_class = len(wdnet_g500)
    print("g500 wdnets", len(wdnet_g500))
    print("w1k wdnets", len(wdnet_w1k))
    print("w1k not in g500 wdnets", len(wdnet_w1k-wdnet_g500))
    root_path = "results/web-g500-metapro-openset"
    root_dirs = [
        "vanilla",
        "wo_rerank_wo_cb",
        "w_rerank_wo_cb",
        "bopro",
        "bopro_mixup",
    ]

    precisions_web = []
    recalls_web = []
    f1s_web = []

    precisions_img = []
    recalls_img = []
    f1s_img = []

    for root_dir in tqdm(root_dirs):
        print("root_dir:", root_dir)
        root_dir_path = os.path.join(root_path, root_dir)
        for csvfile in ["WebVision_predictions.csv", "ImgNet_predictions.csv"]:
            csvfile_path = os.path.join(root_dir_path, csvfile)
            save_txt_path = os.path.join(root_dir_path, "openset_{}_results.txt".format(csvfile))
            precisions = []
            recalls = []
            f1s = []
            with open(save_txt_path, "w") as fw:
                for th_score in tqdm(th_score_list):
                    ### calculate C-P; C-R; C-F1 with respect to G500 categories
                    TP = [0 for _ in range(num_class+1)]
                    FP = [1e-4 for _ in range(num_class+1)]
                    FN = [1e-4 for _ in range(num_class+1)]
                    with open(csvfile_path, "r") as fr:
                        csvreader = csv.reader(fr)
                        for idx, line in enumerate(csvreader):
                            if idx == 0:
                                continue
                            tfrecord,pred_prob,pred_class,target_class = line
                            pred_prob = float(pred_prob)
                            pred_class = int(float(pred_class))
                            target_class = int(float(target_class))
                            pred_wdnet = mapping_idx2wdnet_g500[int(pred_class)]
                            gt_wdnet = mapping_idx2wdnet_web1k[int(target_class)]
                            if gt_wdnet in wdnet_g500:
                                ### in-set
                                gt_class = mapping_wdnet2idx_g500[gt_wdnet]
                                if (float(pred_prob) > th_score):
                                    ## predicted as known class
                                    if pred_wdnet == gt_wdnet:
                                        ## correct
                                        TP[int(pred_class)] += 1
                                    else:
                                        ## incorrect
                                        FP[int(pred_class)] += 1
                                        FN[int(gt_class)] += 1
                                else:
                                    ## predicted as unknown class
                                    FP[num_class] += 1
                                    FN[int(gt_class)] += 1
                            else:
                                ### open-set
                                if (float(pred_prob) > th_score):
                                    FP[int(pred_class)] += 1
                                    FN[num_class] += 1
                                else:
                                    TP[num_class] += 1

                        precision = np.array(TP)/(np.array(TP)+np.array(FP)+1e-4)
                        recall = np.array(TP)/(np.array(TP)+np.array(FN)+1e-4)
                        precision_c = np.mean(precision) * 100
                        recall_c = np.mean(recall) * 100
                        f1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c)
                        # print("{} C-P {} C-R {} C-F1 {}".format(csvfile, precision_c, recall_c, f1_c))
                        # precision_all = np.sum(TP)*100/(np.sum(TP)+np.sum(FP)+1e-4)
                        # recall_all = np.sum(TP)*100/(np.sum(TP)+np.sum(FP)+1e-4)
                        # f1_all = 2*(precision_all*recall_all)/(precision_all+recall_all)
                        # print("{} O-P {} O-R {} O-F1 {}".format(csvfile, precision_all, recall_all, f1_all))
                        precisions.append(precision_c)
                        recalls.append(recall_c)
                        f1s.append(f1_c)
                        
                        fw.write("{} th={} C-P {} C-R {} C-F1 {}\n".format(csvfile,\
                            th_score, precision_c, recall_c, f1_c))
                        # fw.write("{} O-P {} O-R {} O-F1 {}\n".format(csvfile, precision_all, recall_all, f1_all))
            
            if csvfile == "WebVision_predictions.csv":
                precisions_web.append(precisions)
                recalls_web.append(recalls)
                f1s_web.append(f1s)
            else:
                precisions_img.append(precisions)
                recalls_img.append(recalls)
                f1s_img.append(f1s)
    
    np.save(save_npy_path,
        np.array([precisions_web, recalls_web, f1s_web,\
            precisions_img, recalls_img, f1s_img]))

else:
    results = np.load(save_npy_path)
    precisions_web, recalls_web, f1s_web,\
        precisions_img, recalls_img, f1s_img = results

    def calc_precision_recall_auc(precision, recall):
        auc = 0
        assert(len(precision) == len(recall))
        for i in range(len(precision)-1):
            auc += abs(recall[i]-recall[i+1])/100 * precision[i]/100
        return auc

    method_names = ["Vanilla", "Proposed w/o RR & CB", "Proposed w/o CB", "Proposed", "Proposed w/ MU"]
    save_auc = os.path.join(save_dir, "auc.txt")
    with open(save_auc, "w") as fw:
        for i in range(len(precisions_web)):
            precision = precisions_web[i]
            recall = recalls_web[i]
            auc = calc_precision_recall_auc(precision, recall)
            fw.write("WebVision AUC@{}={} \n".format(method_names[i], auc))
        for i in range(len(precisions_img)):
            precision = precisions_img[i]
            recall = recalls_img[i]
            auc = calc_precision_recall_auc(precision, recall)
            fw.write("ImageNet AUC@{}={} \n".format(method_names[i], auc))

        for i in range(len(f1s_web)):
            fw.write("WebVision C-F1 avgs@{}={} \n".format(method_names[i], np.mean(f1s_web[i])))
        for i in range(len(f1s_img)):
            fw.write("ImgNet C-F1 avgs@{}={} \n".format(method_names[i], np.mean(f1s_img[i])))

    fig = plt.figure()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("C-P")
    plt.plot(th_score_list, precisions_web[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(th_score_list, precisions_web[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(th_score_list, precisions_web[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(th_score_list, precisions_web[3], color="purple", linestyle='-', label="Proposed")
    if len(precisions_web) == 5:
        plt.plot(th_score_list, precisions_web[4], color="red", linestyle='-', label="Proposed w/ MU")
    ax.legend()
    ax.set_xlim([0.01, 0.99])
    # ax.set_ylim([50, 70])
    plt.savefig(os.path.join(save_dir, "Precision_webvision_OSR.png"))

    fig = plt.figure()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("C-R")
    plt.plot(th_score_list, recalls_web[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(th_score_list, recalls_web[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(th_score_list, recalls_web[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(th_score_list, recalls_web[3], color="red", linestyle='-', label="Proposed")
    if len(recalls_web) == 5:
        plt.plot(th_score_list, recalls_web[4], color="purple", linestyle='-', label="Proposed w/ MU")
    ax.legend(facecolor="white")
    ax.set_xlim([0.01, 0.99])
    # ax.set_ylim([50, 70])
    plt.savefig(os.path.join(save_dir, "Recall_webvision_OSR.png"))

    fig = plt.figure()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("C-F1")
    plt.plot(th_score_list, f1s_web[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(th_score_list, f1s_web[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(th_score_list, f1s_web[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(th_score_list, f1s_web[3], color="red", linestyle='-', label="Proposed")
    if len(f1s_web) == 5:
        plt.plot(th_score_list, f1s_web[4], color="purple", linestyle='-', label="Proposed w/ MU")
    ax.legend(facecolor="white")
    ax.set_xlim([0.01, 0.99])
    ax.set_ylim([55, 67])
    plt.savefig(os.path.join(save_dir, "F1_webvision_OSR.png"))

    fig = plt.figure()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("C-P")
    plt.plot(th_score_list, precisions_img[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(th_score_list, precisions_img[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(th_score_list, precisions_img[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(th_score_list, precisions_img[3], color="red", linestyle='-', label="Proposed")
    if len(precisions_img) == 5:
        plt.plot(th_score_list, precisions_img[4], color="purple", linestyle='-', label="Proposed w/ MU")
    ax.legend(facecolor="white")
    ax.set_xlim([0.01, 0.99])
    # ax.set_ylim([50, 70])
    plt.savefig(os.path.join(save_dir, "Precision_imgnet_OSR.png"))

    fig = plt.figure()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("C-R")
    plt.plot(th_score_list, recalls_img[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(th_score_list, recalls_img[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(th_score_list, recalls_img[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(th_score_list, recalls_img[3], color="red", linestyle='-', label="Proposed")
    if len(recalls_img) == 5:
        plt.plot(th_score_list, recalls_img[4], color="purple", linestyle='-', label="Proposed w/ MU")
    ax.legend(facecolor="white")
    ax.set_xlim([0.01, 0.99])
    plt.savefig(os.path.join(save_dir, "Recall_imgnet_OSR.png"))

    fig = plt.figure()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("C-F1")
    plt.plot(th_score_list, f1s_img[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(th_score_list, f1s_img[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(th_score_list, f1s_img[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(th_score_list, f1s_img[3], color="red", linestyle='-', label="Proposed")
    if len(f1s_img) == 5:
        plt.plot(th_score_list, f1s_img[4], color="purple", linestyle='-', label="Proposed w/ MU")
    ax.legend(facecolor="white")
    ax.set_xlim([0.01, 0.99])
    ax.set_ylim([52, 62])
    plt.savefig(os.path.join(save_dir, "F1_imgnet_OSR.png"))


    fig = plt.figure()
    ax = plt.axes()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("C-R")
    ax.set_ylabel("C-P")
    plt.plot(recalls_web[0], precisions_web[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(recalls_web[1], precisions_web[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(recalls_web[2], precisions_web[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(recalls_web[3], precisions_web[3], color="purple", linestyle='-', label="Proposed")
    if len(precisions_web) == 5:
        plt.plot(recalls_web[4], precisions_web[4], color="red", linestyle='-', label="Proposed w/ MU")
    ax.legend()
    ax.set_xlim([0, 100])
    # ax.set_ylim([50, 70])
    plt.savefig(os.path.join(save_dir, "PrecisionRecall_webvision_OSR.png"))

    fig = plt.figure()
    ax = plt.axes()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("C-R")
    ax.set_ylabel("C-P")
    plt.plot(recalls_img[0], precisions_img[0], color="green", linestyle='--', label="Vanilla")
    plt.plot(recalls_img[1], precisions_img[1], color="blue", linestyle='-.', label="Proposed w/o RR & CB")
    plt.plot(recalls_img[2], precisions_img[2], color="orange", linestyle=':', label="Proposed w/o CB")
    plt.plot(recalls_img[3], precisions_img[3], color="red", linestyle='-', label="Proposed")
    if len(precisions_img) == 5:
        plt.plot(recalls_img[4], precisions_img[4], color="purple", linestyle='-', label="Proposed w/ MU")
    ax.legend(facecolor="white")
    ax.set_xlim([0, 100])
    # ax.set_ylim([50, 70])
    plt.savefig(os.path.join(save_dir, "PrecisionRecall_imgnet_OSR.png"))

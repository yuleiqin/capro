#!/usr/bin/env python
import argparse
import os
import random
import tensorboard_logger as tb_logger
import json
import csv
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import BoPro, init_weights
import DataLoader.nuswide_dataset as nuswide
import DataLoader.webvision_dataset as webvision
from config_train import parser
import warnings
warnings.filterwarnings('ignore')



def main():
    args = parser.parse_args()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print("=> creating model '{}'".format(args.arch))
    model = BoPro(args)
    if not (args.pretrained):
        model.apply(init_weights)
    model = model.cuda(args.gpu)
    model.eval()

    # resume from a checkpoint
    assert(os.path.exists(args.resume) and os.path.isfile(args.resume)), "must load trained model ckpt for noise cleaning"
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module'):
            # remove prefix
            ## 如果是多卡训练存储ckpt时会加上前缀module
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]        
    model.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    args.distributed = False

    cudnn.benchmark = True
    # Data loading code
    # Data loading code
    assert(os.path.exists(args.root_dir)), "please make sure the path to web data is valid {}".format(args.root_dir)
    if args.use_fewshot:
        assert(os.path.exists(args.root_dir_t)), "please make sure the path to fewshot target domain data is valid {}".format(args.root_dir_t)
        assert(os.path.isfile(args.pathlist_t)), "please make sure the pathlist path to fewshot target domain data is valid {}".format(args.pathlist_t)

    ## load webvision dataset
    assert(os.path.isfile(args.pathlist_web)), "please make sure the pathlist path to webvision web data is valid"
    assert(os.path.exists(args.root_dir_test_web)), "please make sure the path to webvision web test data is valid"
    assert(os.path.isfile(args.pathlist_test_web)), "please make the pathlist path to webvision web test data is valid"
    assert(os.path.exists(args.root_dir_test_target)), "please make sure the path to webvision imgnet test data is valid"
    assert(os.path.isfile(args.pathlist_test_target)), "please make the pathlist path to webvision imgnet test data is valid"
    
    if args.nuswide:
        loader = nuswide.nuswide_dataloader(batch_size=args.batch_size,\
        num_class=args.num_class, num_workers=args.workers,\
            root_dir=args.root_dir, pathlist=args.pathlist_web,\
                root_dir_test_web=args.root_dir_test_web,\
                    pathlist_test_web=args.pathlist_test_web,\
                        root_dir_test_target=args.root_dir_test_target,\
                            pathlist_test_target=args.pathlist_test_target, distributed=args.distributed, crop_size=0.99,\
                                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                                        use_fewshot=args.use_fewshot, annotation="",\
                                            no_color_transform=args.no_color_transform, eval_only=args.eval_only,\
                                                rebalance_downsample=args.rebalance, use_meta_weights=args.use_meta_weights, drop_last=False)
    else:
        loader = webvision.webvision_dataloader(batch_size=args.batch_size,\
            num_class=args.num_class, num_workers=args.workers,\
                root_dir=args.root_dir, pathlist=args.pathlist_web,\
                    root_dir_test_web=args.root_dir_test_web,\
                        pathlist_test_web=args.pathlist_test_web,\
                            root_dir_test_target=args.root_dir_test_target,\
                                pathlist_test_target=args.pathlist_test_target, distributed=args.distributed, crop_size=0.99,\
                                    root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                                        save_dir=args.exp_dir, dry_run=args.dry_run,\
                                            use_fewshot=args.use_fewshot, annotation="",\
                                                no_color_transform=args.no_color_transform, eval_only=args.eval_only,\
                                                    rebalance_downsample=args.rebalance, use_meta_weights=args.use_meta_weights, drop_last=False)
    train_loader, fewshot_loader, test_loader_web, test_loader_target = loader.run()
    
    samples = []
    targets = []
    domains = []
    root_dirs = []
    root_dirs2index = {}
    index2root_dirs = {}
    count_index = 0
    info_all = []
    info_all_json = []

    print("=> performing noise cleaning on the training data")
    with torch.no_grad():   
        for batch in tqdm(train_loader):

            clean_idx, target_org, target, self_prediction, self_prediction_compress,\
                prototype_similarity, soft_label, domain, pathlist = model(batch,\
                    args, is_eval=False, is_analysis=True)

            assert(len(clean_idx) == len(target_org))
            assert(len(target) == len(target_org))
            assert(len(self_prediction) == len(target_org))
            assert(len(self_prediction_compress) == len(target_org))
            assert(len(prototype_similarity) == len(target_org))
            assert(len(soft_label) == len(target_org))
            assert(len(domain) == len(target_org))
            assert(len(pathlist) == len(target_org))

            for clean_i, target_org_i, target_i, predict_i, predict_compress_i,\
                proto_sim_i, soft_label_i, domain_i, pathlist_i in zip(
                    clean_idx.cpu(), target_org.cpu().numpy(),\
                    target.cpu().numpy(), self_prediction.cpu().numpy(),\
                        self_prediction_compress.cpu().numpy(),\
                            prototype_similarity.cpu().numpy(),\
                                soft_label.cpu().numpy(),\
                                    domain.cpu().numpy(), pathlist):
                clean_i = int(clean_i)
                info = pathlist_i.split("@")
                tfrecord = info[0]
                offset = int(info[1])
                img_root_dir = os.path.dirname(tfrecord)
                tfrecord_name = os.path.basename(tfrecord)
                ## 记录文件名仅仅记录底层路径以省略文本
                if not img_root_dir in root_dirs2index:
                    root_dirs2index[img_root_dir] = count_index
                    index2root_dirs[count_index] = img_root_dir
                    count_index += 1
                root_index_i = root_dirs2index[img_root_dir]

                if args.nuswide:
                    ## 移除无效类别元素(赋值为-1)
                    domain_i_nonzero = (np.nonzero(domain_i)[0]).tolist()
                    if len(domain_i_nonzero) == 0:
                        domain_i = 0
                    else:
                        domain_i = 1
                    target_org_i[target_org_i==-1] = 0
                    target_i[target_i==-1] = 0
                    target_org_i = (np.nonzero(target_org_i)[0]).tolist()
                    target_i = (np.nonzero(target_i)[0]).tolist()
                else:
                    domain_i = int(domain_i)
                    target_org_i = [int(target_org_i)]
                    target_i = [int(target_i)]

                target_org = []
                target_org_str = []

                num_pos_tgt_org, num_pos_tgt_corrected = len(target_org_i), len(target_i)
                num_interrected = len(set(target_org_i) & set(target_i))
                
                for class_id in target_org_i:
                    pred_prob_id = float(predict_i[int(class_id)])
                    pred_compress_prob_id = float(predict_compress_i[int(class_id)])
                    proto_id = float(proto_sim_i[int(class_id)])
                    soft_id = float(soft_label_i[int(class_id)])
                    class_str = [str(class_id), "{:.2f}".format(pred_prob_id),
                        "{:.2f}".format(pred_compress_prob_id), "{:.2f}".format(proto_id),
                            "{:.2f}".format(soft_id)]
                    target_org.append(class_str)
                    class_str = ",".join(class_str)
                    target_org_str.append(class_str)

                target_org_str = ";".join(target_org_str)

                target_new = []
                target_new_str = []

                for class_id in target_i:
                    pred_prob_id = float(predict_i[int(class_id)])
                    pred_compress_prob_id = float(predict_compress_i[int(class_id)])
                    proto_id = float(proto_sim_i[int(class_id)])
                    soft_id = float(soft_label_i[int(class_id)])
                    class_str = [str(class_id), "{:.2f}".format(pred_prob_id),
                        "{:.2f}".format(pred_compress_prob_id), "{:.2f}".format(proto_id),
                            "{:.2f}".format(soft_id)]
                    target_new.append(class_str)
                    class_str = ",".join(class_str)
                    target_new_str.append(class_str)
                target_new_str = ";".join(target_new_str)

                info_all.append(["{}@{}".format(tfrecord_name, offset), str(clean_i),
                    target_org_str, target_new_str,
                        str(num_pos_tgt_org), str(num_pos_tgt_corrected), str(num_interrected)])
                info_all_json.append([pathlist_i, target_org, target_new, domain_i])

                if clean_i:
                    samples.append([tfrecord_name, offset])
                    root_dirs.append(int(root_index_i))
                    targets.append(target_i)
                    domains.append(domain_i)
    
    if not args.annotation.endswith(".json"):
        args.annotation = args.annotation + ".json"
    root_dir_anno = os.path.dirname(args.annotation)
    os.makedirs(root_dir_anno, exist_ok=True)
    csv_path = args.annotation.replace(".json", "_all.csv")
    with open(csv_path, "w") as fw:
        csv_writer = csv.writer(fw)
        csv_writer.writerow(["tfrecord_name@offset", "is_clean?",
            "original targets, pred prob, pred prob compress, prototype similarity, soft score",
            "correct targets, pred prob, pred prob compress, prototype similarity, soft score",
            "#positive classes (original)", "#positive classes (corrected)",
            "#interected classes (original & corrected)"])
        for info_i in info_all:
            csv_writer.writerow(info_i)
    json_all_path = args.annotation.replace(".json", "_all.json")
    with open(json_all_path, "w") as fw:
        json.dump(info_all_json, fw)

    with open(args.annotation, "w") as f:
        json.dump({'samples':samples,\
            'targets':targets,\
                'domains':domains,\
                    "roots":root_dirs,\
                        "root2index":root_dirs2index,\
                            "index2root":index2root_dirs}, f)

    print("=> pseudo-label annotation saved to {}".format(args.annotation))
    return


if __name__ == '__main__':
    main()

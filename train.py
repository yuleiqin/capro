## python library
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import json
## pytorch library
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from tqdm import tqdm
from config_train import parser
from model import BoPro, init_weights, concat_all_gather
from utils.metric_multilabel import precision_recall_map, sigmoid_loss_with_posweight
import DataLoader.nuswide_dataset as nuswide
import DataLoader.webvision_dataset as webvision
import tensorboard_logger as tb_logger
from graph.knn_utils import global_build, sgc_precompute
import warnings
warnings.filterwarnings('ignore')



def main():
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print('You have chosen to seed training.')
        # cudnn.deterministic = True
        # warnings.warn('This will turn on the CUDNN deterministic setting, '
        # 'which can slow down your training considerably! '
        # 'You may see unexpected behavior when restarting '
        # 'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("distributed training {}".format(args.distributed))
    ## prepare the directory for saving
    os.makedirs(args.exp_dir, exist_ok=True)
    args.tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    ngpus_per_node = torch.cuda.device_count()
    ## prepare for small batch size
    if args.dry_run:
        args.batch_size = 8
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("{} worldsize for multiprocessing distributed with {} gpus".format(args.world_size,\
            ngpus_per_node))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("{} single process running with {} gpus".format(args.world_size,\
            ngpus_per_node))
        main_worker(args.gpu, ngpus_per_node, args)
    return


def change_status_encoder(model, requires_grad=True):
    """更改encoder参数是否需要梯度更新
    """
    updated_parameters_names = []
    for name_p, p in model.named_parameters():
        if name_p.startswith("module.encoder_q"):
            p.requires_grad = requires_grad
        if p.requires_grad:
            updated_parameters_names.append(name_p)
    params = [p for p in model.parameters() if p.requires_grad]
    print("Set encoder requires_grad = {}".format(requires_grad))
    print("Updated parameter names", updated_parameters_names)
    return model, params


def optimize_encoder(model, args):
    """将encoder部分参数设置为需要梯度更新 同时将其参数加入optimizer进行优化
    """
    model, params = change_status_encoder(model, True)
    ## webvision/google500/nuswide
    optimizer = torch.optim.SGD(params, args.lr,
    momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer


def check_valid_path(args):
    if args.use_fewshot:
        assert(os.path.exists(args.root_dir_t)), "please make sure the path to fewshot target domain data is valid {}".format(args.root_dir_t)
        assert(os.path.isfile(args.pathlist_t)), "please make sure the pathlist path to fewshot target domain data is valid {}".format(args.pathlist_t)
    assert(os.path.exists(args.root_dir)), "please make sure the path to web data is valid"
    assert(os.path.isfile(args.pathlist_web)), "please make sure the pathlist path to webvision web data is valid"
    assert(os.path.exists(args.root_dir_test_web)), "please make sure the path to webvision web test data is valid"
    assert(os.path.isfile(args.pathlist_test_web)), "please make the pathlist path to webvision web test data is valid"
    assert(os.path.exists(args.root_dir_test_target)), "please make sure the path to webvision imgnet test data is valid"
    assert(os.path.isfile(args.pathlist_test_target)), "please make the pathlist path to webvision imgnet test data is valid"
    return


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))    
        
    ## suppress printing if not master
    if not args.dry_run:
        if args.multiprocessing_distributed and args.gpu != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

    print("is_update_proto:", args.update_proto_freq!=0, args.update_proto_freq)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    ## create model
    print("=> creating model '{}'".format(args.arch))
    model = BoPro(args)
    if not args.pretrained:
        ## 如果不使用预训练参数则需要随机初始化
        model.apply(init_weights)

    if args.gpu == 0:
        with open(os.path.join(args.exp_dir, 'commandline_args.txt'), 'w') as fw:
            json.dump(args.__dict__, fw, indent=2)
        print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    ## 优化器选择与学习率衰减策略根据数据集差异有所不同
    if args.frozen_encoder_epoch != 0:
        ## 保证feature encoder backbone不变微调classifier
        model, params = change_status_encoder(model, False)
    else:
        ## all model parameters need to be optimized
        params = model.parameters()

    ## webvision/google500
    optimizer = torch.optim.SGD(params, args.lr,
    momentum=args.momentum, weight_decay=args.weight_decay)
    if not (args.cos):
        for milestone in args.schedule:
            print(milestone)

    def adjust_learning_rate(optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        if args.warmup_epoch != 0 and epoch <= args.warmup_epoch:
            lr *= (max(epoch, 1.0)/args.warmup_epoch)
        else:
            if args.cos:  # cosine lr schedule
                lr *= 0.5 * (1. + math.cos(math.pi * (epoch-args.warmup_epoch)/(args.epochs-args.warmup_epoch)))
            else:
                ## stepwise lr schedule
                for milestone in args.schedule:
                    lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    ## optionally resume from a checkpoint
    resume_path = '{}/checkpoint_latest.tar'.format(args.exp_dir)
    resume_continue = True
    if not os.path.exists(resume_path):
        resume_path = args.resume
        resume_continue = False

    if os.path.exists(resume_path) and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        if args.gpu is None:
            checkpoint = torch.load(resume_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(resume_path, map_location=loc)

        if not resume_continue:
            ## 只能加载预训练参数而不是继续训练
            print("Load parameters to train from scratch")
            model_dict =  model.state_dict()
            state_dict = checkpoint['state_dict']
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict and (model_dict[k].size() == v.size())}
            print("Succesfully Loader Parameters Include", state_dict.keys())
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        else:
            ## 阶段1&阶段3训练
            print("Load parameters to continue training")
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except Exception:
                model_dict =  model.state_dict()
                state_dict = checkpoint['state_dict']
                state_dict = {k:v for k, v in state_dict.items() if k in model_dict and (model_dict[k].size() == v.size())}
                state_dict_mismatch = {k:v for k, v in state_dict.items() if not (k in model_dict and (model_dict[k].size() == v.size()))}
                print("Attention!!! MisMatch parameters!!!", state_dict_mismatch.keys())
                print("Succesfully Loader Parameters Include", state_dict.keys())
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
            if checkpoint['epoch'] >= args.start_epoch and resume_continue:
                ## 仅当checkpoint中的epoch数目大于初始值时才更新
                args.start_epoch = checkpoint['epoch']
            if args.pretrained and args.start_epoch > args.frozen_encoder_epoch:
                ## finetune所有参数, 将optimizer进行替换
                print("optimizer all encoder")
                model, optimizer = optimize_encoder(model, args)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Load optimizer success")
            except:
                print("Load optimizer failed")
    
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))

        if args.nuswide:
            ### 加载的是nuswide数据集对应的评测指标(C-F1; O-F1; mAP)
            if "best_cf1" in checkpoint:
                cf1_max = checkpoint['best_cf1']
            else:
                cf1_max = 0
            
            if "best_of1" in checkpoint:
                of1_max = checkpoint['best_of1']
            else:
                of1_max = 0
            
            if "best_mAP" in checkpoint:
                mAP_max = checkpoint['best_mAP']
            else:
                mAP_max = 0

            if 'epoch_best_cf1' in checkpoint:
                epoch_best_cf1 = checkpoint["epoch_best_cf1"]
            else:
                epoch_best_cf1 = -1

            if 'epoch_best_of1' in checkpoint:
                epoch_best_of1 = checkpoint["epoch_best_of1"]
            else:
                epoch_best_of1 = -1

            if 'epoch_best_mAP' in checkpoint:
                epoch_best_mAP = checkpoint["epoch_best_mAP"]
            else:
                epoch_best_mAP = -1

        else:
            ### 加载的是webvision数据集对应的评测指标(best acc imgnet; best acc webvision)
            if ("best_acc_web" in checkpoint) and (type(checkpoint["best_acc_web"]) is list):
                acc_max_web1, acc_max_web5 = checkpoint["best_acc_web"]
            else:
                acc_max_web1, acc_max_web5 = 0, 0

            if ("best_acc_imgnet" in checkpoint) and (type(checkpoint["best_acc_imgnet"]) is list):
                acc_max_imgnet1, acc_max_imgnet5 = checkpoint["best_acc_imgnet"]
            else:
                acc_max_imgnet1, acc_max_imgnet5 = 0, 0

            if 'epoch_best_web' in checkpoint:
                epoch_best_web = checkpoint["epoch_best_web"]
            else:
                epoch_best_web = -1

            if 'epoch_best_imgnet' in checkpoint:
                epoch_best_imgnet = checkpoint["epoch_best_imgnet"]
            else:
                epoch_best_imgnet = -1
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))
        if args.nuswide:
            ### 加载的是nuswide数据集对应的评测指标
            cf1_max, of1_max, mAP_max = 0, 0, 0
            epoch_best_cf1, epoch_best_of1, epoch_best_mAP = -1, -1, -1
        else:
            ### 加载的是webvision数据集对应的评测指标
            acc_max_web1, acc_max_web5, acc_max_imgnet1, acc_max_imgnet5 = 0, 0, 0, 0
            epoch_best_web, epoch_best_imgnet = -1, -1

    cudnn.benchmark = True
    # Data loading code
    check_valid_path(args)
    if args.nuswide:
        loader = nuswide.nuswide_dataloader(batch_size=args.batch_size,\
        num_class=args.num_class, num_workers=args.workers,\
            root_dir=args.root_dir, pathlist=args.pathlist_web,\
                root_dir_test_web=args.root_dir_test_web,\
                    pathlist_test_web=args.pathlist_test_web,\
                        root_dir_test_target=args.root_dir_test_target,\
                            pathlist_test_target=args.pathlist_test_target, distributed=args.distributed, crop_size=0.8,\
                                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                                        use_fewshot=args.use_fewshot, annotation=args.annotation,\
                                            no_color_transform=args.no_color_transform, eval_only=args.eval_only,\
                                                rebalance_downsample=args.rebalance, use_meta_weights=args.use_meta_weights, topk=args.topk)
    else:
        loader = webvision.webvision_dataloader(batch_size=args.batch_size,\
            num_class=args.num_class, num_workers=args.workers,\
                root_dir=args.root_dir, pathlist=args.pathlist_web,\
                    root_dir_test_web=args.root_dir_test_web,\
                        pathlist_test_web=args.pathlist_test_web,\
                            root_dir_test_target=args.root_dir_test_target,\
                                pathlist_test_target=args.pathlist_test_target, distributed=args.distributed, crop_size=0.2,\
                                    root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                                        save_dir=args.exp_dir, dry_run=args.dry_run,\
                                            use_fewshot=args.use_fewshot, annotation=args.annotation,\
                                                no_color_transform=args.no_color_transform, eval_only=args.eval_only,\
                                                    rebalance_downsample=args.rebalance, use_meta_weights=args.use_meta_weights, topk=args.topk)
    train_loader, fewshot_loader, test_loader_web, test_loader_target = loader.run()

    if args.nuswide or args.multi_label:
        pos_weight = train_loader.dataset.pos_weights
        criterion = nn.BCEWithLogitsLoss(reduction='none',\
            pos_weight=torch.Tensor(pos_weight).cuda(args.gpu)).cuda(args.gpu)
        criterion_pcl = nn.CrossEntropyLoss(reduction='none',\
            weight=torch.Tensor([1, np.mean(pos_weight)]).cuda(args.gpu)).cuda(args.gpu)
        criterion_ce = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)

    if args.gpu==0:
        logger = tb_logger.Logger(logdir=args.tensorboard_dir, flush_secs=2)
    else:
        logger = None
    
    if args.eval_only:
        eval_only(model, test_loader_web, test_loader_target, args, args.start_epoch, logger)
        return
    
    print("=> start training from epoch {}".format(args.start_epoch))
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            loader.train_sampler.set_epoch(epoch)
        ## set model learning rate
        lr_epoch = adjust_learning_rate(optimizer, epoch, args)
        ## initialize prototype features
        if (epoch == args.init_proto_epoch) or (args.update_proto_freq == 0) or (args.dry_run):
            init_prototype_fewshot(model, fewshot_loader, args, epoch)
            ## save checkpoint model
            if args.rank % ngpus_per_node == 0:
                if args.nuswide:
                    save_checkpoint_nuswide(epoch + 1, args.arch, model, optimizer,\
                        cf1_max, of1_max, mAP_max, epoch_best_cf1,\
                            epoch_best_of1, epoch_best_mAP, '{}/checkpoint_init_proto.tar'.format(args.exp_dir))
                else:
                    save_checkpoint_webvision(epoch + 1, args.arch, model, optimizer,
                        acc_max_web1, acc_max_web5, acc_max_imgnet1, acc_max_imgnet5,\
                            epoch_best_web, epoch_best_imgnet, '{}/checkpoint_init_proto.tar'.format(args.exp_dir))

        if args.nuswide:
            ## train/val with nuswide
            train_nuswide(train_loader, model, criterion, criterion_ce, criterion_pcl,\
                optimizer, epoch, args, logger, lr_epoch)
            ## save the latest checkpoint model
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint_nuswide(epoch + 1, args.arch, model, optimizer,\
                    cf1_max, of1_max, mAP_max, epoch_best_cf1,\
                        epoch_best_of1, epoch_best_mAP, '{}/checkpoint_latest.tar'.format(args.exp_dir))
            ## test NUSWide dataset
            cf1, of1, mAP = test_nuswide(model, test_loader_web, args, epoch, logger, dataset_name="NUSWide")
            ## save the best checkpoint model
            if cf1 > cf1_max:
                cf1_max = cf1
                epoch_best_cf1 = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint_nuswide(epoch + 1, args.arch, model, optimizer,\
                        cf1_max, of1_max, mAP_max, epoch_best_cf1,\
                            epoch_best_of1, epoch_best_mAP, '{}/checkpoint_best_cf1.tar'.format(args.exp_dir))
            if of1 > of1_max:
                of1_max = of1
                epoch_best_of1 = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint_nuswide(epoch + 1, args.arch, model, optimizer,\
                        cf1_max, of1_max, mAP_max, epoch_best_cf1,\
                            epoch_best_of1, epoch_best_mAP, '{}/checkpoint_best_of1.tar'.format(args.exp_dir))
            if mAP > mAP_max:
                mAP_max = mAP
                epoch_best_mAP = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint_nuswide(epoch + 1, args.arch, model, optimizer,\
                        cf1_max, of1_max, mAP_max, epoch_best_cf1,\
                            epoch_best_of1, epoch_best_mAP, '{}/checkpoint_best_mAP.tar'.format(args.exp_dir))

            print("CF1 best = {:.2f} @epoch {}".format(cf1_max, epoch_best_cf1))
            print("OF1 best = {:.2f} @epoch {}".format(of1_max, epoch_best_of1))
            print("mAP best = {:.2f} @epoch {}".format(mAP_max, epoch_best_mAP))
            
        else:
            ## train
            train_webvision(train_loader, model, criterion,\
                optimizer, epoch, args, logger, lr_epoch)
            ## save the latest checkpoint model
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint_webvision(epoch, args.arch, model, optimizer,
                    acc_max_web1, acc_max_web5, acc_max_imgnet1, acc_max_imgnet5,\
                        epoch_best_web, epoch_best_imgnet, '{}/checkpoint_latest.tar'.format(args.exp_dir))
            ## test webvision & imgnet dataset
            acc1_web, acc5_web = test_webvision(model, test_loader_web, args, epoch, logger, dataset_name="WebVision")
            acc1_imgnet, acc5_imgnet = test_webvision(model, test_loader_target, args, epoch, logger, dataset_name="ImgNet")
            ## save the best checkpoint model
            if acc1_web > acc_max_web1:
                acc_max_web1, acc_max_web5 = acc1_web, acc5_web
                epoch_best_web = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint_webvision(epoch, args.arch, model, optimizer,
                        acc_max_web1, acc_max_web5, acc_max_imgnet1, acc_max_imgnet5,\
                            epoch_best_web, epoch_best_imgnet, '{}/checkpoint_best_web.tar'.format(args.exp_dir))
            if acc1_imgnet > acc_max_imgnet1:
                acc_max_imgnet1, acc_max_imgnet5 = acc1_imgnet, acc5_imgnet
                epoch_best_imgnet = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint_webvision(epoch, args.arch, model, optimizer,
                        acc_max_web1, acc_max_web5, acc_max_imgnet1, acc_max_imgnet5,\
                            epoch_best_web, epoch_best_imgnet, '{}/checkpoint_best_imgnet.tar'.format(args.exp_dir))
            
            print("webvision best top1 = {:.2f} top5 = {:.2f} @epoch {}".format(acc_max_web1, acc_max_web5, epoch_best_web))
            print("imagenet best top1 = {:.2f} top5 = {:.2f} @epoch {}".format(acc_max_imgnet1, acc_max_imgnet5, epoch_best_imgnet))
    return


def eval_only(model, test_loader_web, test_loader_target, args, epoch, logger):
    """仅用于两个数据集的测试
    """
    if args.nuswide:
        cf1, of1, mAP = test_nuswide(model, test_loader_web, args, epoch, logger,\
            dataset_name="NUSWide", save_feat=args.save_feat)
    else:
        acc1_web, acc5_web = test_webvision(model, test_loader_web, args, epoch, logger,\
            dataset_name="WebVision", save_feat=args.save_feat)
        acc1_imgnet, acc5_imgnet = test_webvision(model, test_loader_target, args, epoch, logger,\
            dataset_name="ImgNet", save_feat=args.save_feat)
    return


def init_prototype_fewshot(model, fewshot_loader, args, epoch, is_eval=False):
    with torch.no_grad():
        print('==> Initialize FewShot Prototype...[Epoch {}]'.format(epoch))     
        model.eval()
        ## 初始化
        if args.distributed:
            model.module._zero_prototype_features()
        else:
            model._zero_prototype_features()
        ## 按类累加 (此时不归一化)
        for batch in tqdm(fewshot_loader):
            model(batch, args, is_eval=is_eval, is_proto_init=True)
        ## 平均并归一化
        if args.distributed:
            model.module._initialize_prototype_features()
        else:
            model._initialize_prototype_features()
        if not is_eval:
            dist.barrier()
    return


def train_webvision(train_loader, model, criterion, optimizer,\
    epoch, args, tb_logger, lr_epoch):
    if args.rebalance:
        ## 对样本进行重采样并选取固定大小
        train_loader.dataset.resample()

    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')   
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    acc_inst = AverageMeter('Acc@Inst', ':2.2f')
    acc_cls_lowdim = AverageMeter('Acc@Cls_lowdim', ':2.2f')
    mse_reconstruct = AverageMeter('Mse@Reconstruct', ':2.2f')
    acc_relation = AverageMeter('Acc@Relation', ':2.2f')
    acct_relation = AverageMeter('AccTarget@Relation', ':2.2f')
    loss_sum_scalar = AverageMeter('Loss@Sum', ':2.2f')
    loss_cls_scalar = AverageMeter('Loss@Cls', ':2.2f')
    loss_cls_lowdim_scalar = AverageMeter('Loss@Cls_lowdim', ':2.2f')
    loss_proto_scalar = AverageMeter('Loss@Proto', ':2.2f')
    loss_inst_scalar = AverageMeter('Loss@Inst', ':2.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_inst, acc_proto,\
            mse_reconstruct, acc_relation, acct_relation, acc_cls_lowdim,\
                loss_sum_scalar, loss_cls_scalar, loss_cls_lowdim_scalar,\
                    loss_proto_scalar, loss_inst_scalar],

        prefix="Epoch: [{}]".format(epoch))
    print('==> Training...')
    # switch to train mode
    model.train()
    end = time.time()

    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0
        cls_out, target, q_reconstruct, q, q_compress,\
            sample_weight, target_meta_soft, cls_out_compress,\
                inst_logits, inst_labels, logits_proto,\
                    triplet_mixup, target_queue,\
                        triplet_ncr = model(batch, args,\
                            is_proto=(epoch>args.init_proto_epoch),\
                                is_clean=(epoch>args.start_clean_epoch),\
                                    is_update_proto=((args.update_proto_freq!=0) and (epoch%args.update_proto_freq==0)))
        ## classification loss
        if args.use_soft_label in [0, 7, 10]:
            loss_cls = (criterion(cls_out, target)*sample_weight).mean()
        elif args.use_soft_label == 1:
            ## bootstrapping
            loss_cls_hard = criterion(cls_out, target)*sample_weight
            target_bootstrap = (F.softmax(cls_out, dim=1)).detach().clone()
            loss_cls_soft = -torch.sum(target_bootstrap*F.log_softmax(cls_out, dim=1),
                dim=1)*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
        elif args.use_soft_label == 2:
            ## label smoothing
            loss_cls_hard = criterion(cls_out, target)*sample_weight
            loss_cls_soft = -torch.sum((1/args.num_class)*F.log_softmax(cls_out, dim=1),
                dim=1)*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
        elif (args.use_soft_label == 3 or args.use_soft_label == 4):
            ## self-contained confidence and plus
            if args.use_soft_label == 4 and (epoch>args.init_proto_epoch):
                soft_label = args.alpha*F.softmax(cls_out, dim=1) + (1-args.alpha)*F.softmax(logits_proto, dim=1)
            else:
                soft_label = F.softmax(cls_out, dim=1)
            gt_score = soft_label[target>=0,target]
            gt_score = gt_score.detach().clone()
            target_scc = (F.softmax(cls_out, dim=1)).detach().clone()
            loss_cls_hard = criterion(cls_out, target)*sample_weight
            loss_cls_soft = -torch.sum(target_scc*F.log_softmax(cls_out, dim=1),
                dim=1)*sample_weight
            loss_cls = (gt_score*loss_cls_hard).mean() + ((1-gt_score)*loss_cls_soft).mean()
        elif args.use_soft_label == 5:
            ## NCR=Neighborhood Consistency Regularization
            loss_cls_hard = criterion(cls_out, target)*sample_weight
            # cls_out_gather = concat_all_gather(cls_out)
            # sample_weight_gather = (concat_all_gather(sample_weight)).detach().clone()
            # q_gather = (concat_all_gather(q)).detach().clone()
            # q_gather = F.normalize(q_gather, p=2, dim=1)
            # with torch.no_grad():
            #     target_knn_gather = (F.softmax(cls_out_gather/2, dim=1)).detach().clone()
            #     ###use explicit KNN smoothing
            #     ###cuda multigpu gpu cpu
            #     # knn_graph = global_build(feature_root=q_gather, dist_def="cosine:",\
            #     #     k=10, save_filename="", build_graph_method='cuda')
            #     ###use dot similarity KNN smoothing
            #     q_sim = torch.einsum('nc,ck->nk', [q_gather, q_gather.t()])
            #     q_sim = torch.clamp(q_sim, min=0, max=1)+1e-7
            #     sims, ners = q_sim.topk(k=10, dim=1, largest=True)
            #     ###use explicit knn graph text feature smoothing 
            #     # knn_graph = torch.cat([ners.unsqueeze(1), sims.unsqueeze(1)], dim=1)
            #     # target_knn_gather = torch.clamp(sgc_precompute(target_knn_gather,\
            #     #     knn_graph, self_weight=0, edge_weight=True, degree=1), min=0, max=1)+1e-7
            #     target_knn_gather_iter = []
            #     for sim, ner in zip(sims, ners):
            #         target_knn_gather_i = torch.index_select(target_knn_gather, dim=0, index=ner)
            #         sim = sim/(torch.sum(sim)+1e-7)
            #         target_knn_gather_i = (target_knn_gather_i*sim.view(-1, 1)).sum(dim=0)
            #         target_knn_gather_iter.append(target_knn_gather_i.unsqueeze(0))
            #     target_knn_gather = torch.cat(target_knn_gather_iter, dim=0)
            #     target_knn_gather = target_knn_gather/torch.sum(target_knn_gather, dim=1, keepdim=True)
            cls_out_gather, target_knn_gather, sample_weight_gather = triplet_ncr
            loss_cls_soft = -torch.sum(target_knn_gather*F.log_softmax(cls_out_gather, dim=1),
                dim=1)*sample_weight_gather
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + (args.beta)*loss_cls_soft.mean()
        elif args.use_soft_label == 6:
            ## proto-similarity-consistency smoothness
            loss_cls_hard = criterion(cls_out, target)*sample_weight
            if epoch>args.init_proto_epoch:
                target_proto_sim = (F.softmax(logits_proto, dim=1)).detach().clone()
                loss_cls_soft = -torch.sum(target_proto_sim*F.log_softmax(cls_out, dim=1), dim=1)*sample_weight
                loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
            else:
                loss_cls = loss_cls_hard.mean()
        elif args.use_soft_label == 8:
            ## bootstrapping-lowdim
            loss_cls_hard = criterion(cls_out, target)*sample_weight
            target_bootstrap_lowdim = (F.softmax(cls_out_compress, dim=1)).detach().clone()
            loss_cls_soft = -torch.sum(target_bootstrap_lowdim*F.log_softmax(cls_out, dim=1),
                dim=1)*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
        elif args.use_soft_label == 9:
            ## low dimensional collective bootstrapping but enforce on classifier
            loss_cls_hard = criterion(cls_out, target)*sample_weight
            loss_cls_soft = -torch.sum(target_queue*F.log_softmax(cls_out, dim=1),
                dim=1)*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()

        if args.mixup:
            lam_mixup, output_mixup, target_mixup = triplet_mixup
            loss_cls_mixup1 = criterion(output_mixup, target)*sample_weight
            loss_cls_mixup2 = criterion(output_mixup, target_mixup)*sample_weight
            loss_cls_mixup = lam_mixup*loss_cls_mixup1.mean() + (1-lam_mixup)*loss_cls_mixup2.mean()
            if args.use_soft_label == 0:
                loss_cls *= 0  ## 替换原有分类损失
            loss_cls += loss_cls_mixup
        loss_cls_scalar.update(loss_cls.item())
        loss += loss_cls
        acc_cls.update(accuracy(cls_out, target)[0][0])

        ## instance contrastive loss
        loss_inst = (criterion(inst_logits, inst_labels)*sample_weight).mean()
        loss_inst_scalar.update(loss_inst.item())
        loss += args.w_inst*loss_inst
        acc_inst.update(accuracy(inst_logits, inst_labels)[0][0])

        ## low-dim classification loss
        loss_cls_lowdim = (criterion(cls_out_compress, target)*sample_weight).mean()
        if args.use_soft_label in [7, 10]:
            loss_cls_lowdim *= (1-args.beta)
            if args.use_soft_label == 7:
                ## low-dimensional collective bootstrapping
                loss_cls_soft_lowdim = -torch.sum(target_queue*F.log_softmax(cls_out_compress,\
                    dim=1), dim=1)*sample_weight
            elif args.use_soft_label == 10:
                ## low-dimensional bootstrapping
                target_bootstrap_lowdim = (F.softmax(cls_out_compress, dim=1)).detach().clone()
                loss_cls_soft_lowdim = -torch.sum(target_bootstrap_lowdim*F.log_softmax(cls_out_compress,\
                    dim=1), dim=1)*sample_weight
            loss_cls_lowdim += args.beta*loss_cls_soft_lowdim.mean()
        loss += loss_cls_lowdim
        loss_cls_lowdim_scalar.update(loss_cls_lowdim.item())
        acc_cls_lowdim.update(accuracy(cls_out_compress, target)[0][0])

        ## reconstruction loss
        loss_reconstruct = F.mse_loss(q_reconstruct, q.detach().clone())
        loss += args.w_recn*loss_reconstruct
        mse_reconstruct.update(loss_reconstruct.item())
 
        ## prototypical contrastive loss
        if epoch > args.init_proto_epoch:
            loss_proto = (criterion(logits_proto, target)*sample_weight).mean()
            loss_proto_scalar.update(loss_proto.item())
            loss += args.w_proto*loss_proto
            acc_proto.update(accuracy(logits_proto, target)[0][0])

        ## compute gradient and do SGD step
        loss_sum_scalar.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.gpu == 0 and i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Instance Acc', acc_inst.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Reconstruction Mse', mse_reconstruct.avg, epoch)
        tb_logger.log_value('Relation Acc Target', acct_relation.avg, epoch)
        tb_logger.log_value('Relation Acc', acc_relation.avg, epoch)
        tb_logger.log_value('Train LowDim Acc', acc_cls_lowdim.avg, epoch)
 
        tb_logger.log_value('Loss Sum', loss_sum_scalar.avg, epoch)
        tb_logger.log_value('Loss Cls', loss_cls_scalar.avg, epoch)
        tb_logger.log_value('Loss Cls lowdim', loss_cls_lowdim_scalar.avg, epoch)
        tb_logger.log_value('Loss Proto', loss_proto_scalar.avg, epoch)
        tb_logger.log_value('Loss Inst', loss_inst_scalar.avg, epoch)

        tb_logger.log_value('Learning Rate', lr_epoch, epoch)
    return


def train_nuswide(train_loader, model, criterion, criterion_ce, criterion_pcl,\
    optimizer, epoch, args, tb_logger, lr_epoch):
    ### train nuswide dataset
    if args.rebalance:
        ## 对样本进行重采样并选取固定大小
        train_loader.dataset.resample()

    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')   
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    acc_inst = AverageMeter('Acc@Inst', ':2.2f')
    acc_cls_lowdim = AverageMeter('Acc@Cls_lowdim', ':2.2f')
    mse_reconstruct = AverageMeter('Mse@Reconstruct', ':2.2f')
    acc_relation = AverageMeter('Acc@Relation', ':2.2f')
    acct_relation = AverageMeter('AccTarget@Relation', ':2.2f')
    loss_sum_scalar = AverageMeter('Loss@Sum', ':2.2f')
    loss_cls_scalar = AverageMeter('Loss@Cls', ':2.2f')
    loss_cls_lowdim_scalar = AverageMeter('Loss@Cls_lowdim', ':2.2f')
    loss_proto_scalar = AverageMeter('Loss@Proto', ':2.2f')
    loss_inst_scalar = AverageMeter('Loss@Inst', ':2.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_inst, acc_proto,\
            mse_reconstruct, acc_relation, acct_relation, acc_cls_lowdim,\
                loss_sum_scalar, loss_cls_scalar, loss_cls_lowdim_scalar,\
                    loss_proto_scalar, loss_inst_scalar],
        prefix="Epoch: [{}]".format(epoch))
    print('==> Training...')
    # switch to train mode
    model.train()
    end = time.time()

    predictions = []
    predictions_prob_all = []
    targets = []
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0
        cls_out, target, q_reconstruct, q, q_compress,\
            sample_weight, target_meta_soft, cls_out_compress,\
                inst_logits, inst_labels, logits_proto,\
                    triplet_mixup, target_queue = model(batch, args,\
                        is_proto=(epoch>args.init_proto_epoch),\
                            is_clean=(epoch>args.start_clean_epoch),\
                                is_update_proto=((args.update_proto_freq!=0) and (epoch%args.update_proto_freq==0)))
        target_valid = (target>=0).float()
        target[target<0] = 0  # 仅仅是为了防止计算错误
        ## classification loss
        if args.use_soft_label in [0, 7, 10]:
            loss_cls = (criterion(cls_out, target)*target_valid*sample_weight).mean()
        elif args.use_soft_label == 1:
            ## bootstrapping
            loss_cls_hard = criterion(cls_out, target)*target_valid*sample_weight
            target_bootstrap = (torch.sigmoid(cls_out)).detach().clone()
            loss_cls_soft = -target_bootstrap*F.logsigmoid(cls_out)*target_valid*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
        elif args.use_soft_label == 2:
            ## label smoothing
            loss_cls_hard = criterion(cls_out, target)*target_valid*sample_weight
            loss_cls_soft = -(1/2)*F.logsigmoid(cls_out)*target_valid*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
        elif (args.use_soft_label == 3 or args.use_soft_label == 4):
            ## self-contained confidence and plus
            if args.use_soft_label == 4 and (epoch>args.init_proto_epoch):
                gt_score = args.alpha*torch.sigmoid(cls_out) + (1-args.alpha)*F.softmax(logits_proto, dim=2)[...,1]
            else:
                gt_score = torch.sigmoid(cls_out)
            gt_score = gt_score.detach().clone()
            target_scc = (torch.sigmoid(cls_out)).detach().clone()
            loss_cls_hard = criterion(cls_out, target)*target_valid*sample_weight
            loss_cls_soft = -target_scc*F.logsigmoid(cls_out)*target_valid*sample_weight
            loss_cls = (gt_score*loss_cls_hard).mean() + ((1-gt_score)*loss_cls_soft).mean()
        elif args.use_soft_label == 5:
            ## NCR=Neighborhood Consistency Regularization
            cls_out_gather = concat_all_gather(cls_out)
            sample_weight_gather = concat_all_gather(sample_weight)
            target_valid_gather = concat_all_gather(target_valid)
            with torch.no_grad():
                q_gather = F.normalize(concat_all_gather(q), p=2, dim=1)
                knn_graph = global_build(feature_root=q_gather, dist_def="cosine:",\
                    k=min(10, q_gather.size(0)), save_filename="", build_graph_method='cuda')
                target_knn_gather = (torch.sigmoid(cls_out_gather)).detach().clone()
                target_knn_gather = torch.clamp(sgc_precompute(target_knn_gather,\
                    knn_graph, self_weight=0, edge_weight=True, degree=1), min=0, max=1)+1e-7
            loss_cls_hard = criterion(cls_out, target)*target_valid*sample_weight
            loss_cls_soft = -target_knn_gather*F.logsigmoid(cls_out_gather)*target_valid_gather*sample_weight_gather
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + (args.beta/args.world_size)*loss_cls_soft.mean()
        elif args.use_soft_label == 6:
            ## proto-consistency smoothness
            loss_cls_hard = criterion(cls_out, target)*target_valid*sample_weight
            if epoch>args.init_proto_epoch:
                target_proto_sim = (F.softmax(logits_proto, dim=2)[...,1]).detach().clone()
                loss_cls_soft = -target_proto_sim*F.logsigmoid(cls_out)*target_valid*sample_weight
                loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
            else:
                loss_cls = loss_cls_hard.mean()
        elif args.use_soft_label == 8:
            ## bootstrapping-lowdim
            loss_cls_hard = criterion(cls_out, target)*target_valid*sample_weight
            target_bootstrap = (torch.sigmoid(cls_out_compress)).detach().clone()
            loss_cls_soft = -target_bootstrap*F.logsigmoid(cls_out)*target_valid*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()
        elif args.use_soft_label == 9:
            ## low dimensional collective bootstrapping but enforce on classifier
            loss_cls_hard = criterion(cls_out, target)*target_valid*sample_weight
            loss_cls_soft = -target_queue*F.logsigmoid(cls_out)*target_valid*sample_weight
            loss_cls = (1-args.beta)*loss_cls_hard.mean() + args.beta*loss_cls_soft.mean()

        if args.mixup:
            lam_mixup, output_mixup, target_mixup = triplet_mixup
            target_mixup_valid = (target_mixup>=0).float()
            target_mixup[target_mixup<0] = 0  # 仅仅是为了防止计算错误
            loss_cls_mixup1 = criterion(output_mixup, target)*target_valid*sample_weight
            loss_cls_mixup2 = criterion(output_mixup, target_mixup)*target_mixup_valid*sample_weight
            loss_cls_mixup = lam_mixup*loss_cls_mixup1.mean() + (1-lam_mixup)*loss_cls_mixup2.mean()
            if args.use_soft_label == 0:
                loss_cls *= 0  ## 替换原有分类损失
            loss_cls += loss_cls_mixup
        loss_cls_scalar.update(loss_cls.item())
        loss += loss_cls
        acc_cls.update(accuracy_multi_label(torch.sigmoid(cls_out), target, target_valid=target_valid.bool()))

        ## low-dim classification loss
        loss_cls_lowdim = (criterion(cls_out_compress, target)*target_valid*sample_weight).mean()
        if args.use_soft_label in [7, 10]:
            loss_cls_lowdim *= (1-args.beta)
            if args.use_soft_label == 7:
                ## low-dimensional collective bootstrapping
                loss_cls_soft_lowdim = -target_queue*F.logsigmoid(cls_out_compress)*target_valid*sample_weight
            elif args.use_soft_label == 10:
                ## low-dimensional bootstrapping
                target_bootstrap_lowdim = (torch.sigmoid(cls_out)).detach().clone()
                loss_cls_soft_lowdim = -target_bootstrap_lowdim*F.logsigmoid(cls_out_compress)*target_valid*sample_weight
            loss_cls_lowdim += args.beta*loss_cls_soft_lowdim.mean()
        loss += loss_cls_lowdim
        loss_cls_lowdim_scalar.update(loss_cls_lowdim.item())
        acc_cls_lowdim.update(accuracy_multi_label(torch.sigmoid(cls_out_compress), target, target_valid=target_valid.bool()))

        ## reconstruction loss
        loss_reconstruct = F.mse_loss(q_reconstruct, q.detach().clone())
        loss += args.w_recn*loss_reconstruct
        mse_reconstruct.update(loss_reconstruct.item())

        ## instance contrastive loss
        loss_inst = (criterion_ce(inst_logits, inst_labels)*sample_weight).mean()
        loss_inst_scalar.update(loss_inst.item())
        loss += args.w_inst*loss_inst
        acc_inst.update(accuracy(inst_logits, inst_labels)[0][0])

        ## prototypical contrastive loss
        if epoch > args.init_proto_epoch:
            batch_size, num_class = logits_proto.size(0), logits_proto.size(1)
            loss_proto = criterion_pcl(logits_proto.view(batch_size*num_class, 2),\
                target.view(-1).long())*target_valid.view(-1)
            loss_proto = (loss_proto.view(batch_size, num_class)*sample_weight).mean()
            acc_proto.update(accuracy_multi_label(F.softmax(logits_proto, dim=2)[...,1],\
                target, target_valid=target_valid.bool()))
            loss_proto_scalar.update(loss_proto.item())
            loss += args.w_proto*loss_proto

        ## compute gradient and do SGD step
        loss_sum_scalar.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.gpu == 0 and i % args.print_freq == 0:
            progress.display(i)

        if len(cls_out) > 0:
            _, pred = torch.sigmoid(cls_out).topk(3, 1, True, True)
            predictions.append(pred)
            predictions_prob_all.append(torch.sigmoid(cls_out))
            targets.append(target)

    predictions = torch.cat(predictions, dim=0).cuda(args.gpu)
    predictions_prob_all = torch.cat(predictions_prob_all, dim=0).cuda(args.gpu)
    targets = torch.cat(targets, dim=0).cuda(args.gpu)
    ## collect from all gpus
    predictions = concat_all_gather(predictions)
    predictions_prob_all = concat_all_gather(predictions_prob_all)
    targets = concat_all_gather(targets)
    ## from tensor to array
    predictions_npy = predictions.detach().cpu().numpy()
    predictions_prob_all_npy = predictions_prob_all.detach().cpu().numpy()
    targets_npy = targets.detach().cpu().numpy()

    c_precision, c_recall, c_f1,\
        o_precision, o_recall, o_f1, mAP = precision_recall_map(predictions_npy,\
            predictions_prob_all_npy, targets_npy, args.num_class)

    if args.gpu == 0:
        print('Train Epoch %d C-F1 is %.2f%% O-F1 is %.2f%% mAP is %.2f%%'%(int(epoch), c_f1, o_f1, mAP))
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Instance Acc', acc_inst.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Reconstruction Mse', mse_reconstruct.avg, epoch)
        tb_logger.log_value('Relation Acc Target', acct_relation.avg, epoch)
        tb_logger.log_value('Relation Acc', acc_relation.avg, epoch)
        tb_logger.log_value('Train LowDim Acc', acc_cls_lowdim.avg, epoch)

        tb_logger.log_value('Loss Sum', loss_sum_scalar.avg, epoch)
        tb_logger.log_value('Loss Cls', loss_cls_scalar.avg, epoch)
        tb_logger.log_value('Loss Cls lowdim', loss_cls_lowdim_scalar.avg, epoch)
        tb_logger.log_value('Loss Proto', loss_proto_scalar.avg, epoch)
        tb_logger.log_value('Loss Inst', loss_inst_scalar.avg, epoch)
        
        tb_logger.log_value('Train C-Precision', c_precision, epoch)
        tb_logger.log_value('Train C-Recall', c_recall, epoch)
        tb_logger.log_value('Train C-F1', c_f1, epoch)
        tb_logger.log_value('Train O-Precision', o_precision, epoch)
        tb_logger.log_value('Train O-Recall', o_recall, epoch)
        tb_logger.log_value('Train O-F1', o_f1, epoch)
        tb_logger.log_value('Train mAP', mAP, epoch)
        tb_logger.log_value('Learning Rate', lr_epoch, epoch)
    return


def test_webvision(model, test_loader, args, epoch, tb_logger,\
    dataset_name="WebVision", save_feat=False):
    pred_info = []
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        # evaluate on webvision val set
        batch_idx = 0
        for batch in tqdm(test_loader):
            ## outputs, feat, target, feat_reconstruct
            if save_feat:
                ## concatenate all features
                features_all = []
                tfrecord_names_batch = []
            tfrecord_names = batch[3]
            outputs, target, _, q = model(batch, args, is_eval=True)
            if args.multi_label:
                target = target.max(dim=1)[1]
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])
            if args.eval_only:
                pred_prob, pred_id = torch.max(F.softmax(outputs, dim=1), dim=1)
                for tfrecord_name, pred_prob_i,\
                    pred_id_i, target_i in zip(tfrecord_names, pred_prob, pred_id, target):
                    pred_info.append([tfrecord_name, float(pred_prob_i), float(pred_id_i), int(target_i)])
            
            if save_feat:
                assert(len(tfrecord_names) == len(q))
                for tfrecord_name, feature in zip(tfrecord_names, q):
                    features_all.append(feature.detach().cpu().numpy().tolist())
                    tfrecord_names_batch.append(str(tfrecord_name))
                if args.gpu ==0:
                    save_names_path = os.path.join(args.exp_dir, "{}_tfrecord_names_all_{}.txt".format(dataset_name, batch_idx))
                    save_feat_path = os.path.join(args.exp_dir, "{}_save_feats_{}.npy".format(dataset_name, batch_idx))
                    np.save(save_feat_path, np.array(features_all))
                    with open(save_names_path, "w") as fw:
                        for tfrecord_name in tfrecord_names_batch:
                            fw.write(tfrecord_name + "\n")
            batch_idx += 1
        
        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        dist.all_reduce(acc_tensors)        
        acc_tensors /= args.world_size

        if args.eval_only:
            import csv
            save_csv_file = os.path.join(args.exp_dir, "{}_predictions.csv".format(dataset_name))
            with open(save_csv_file, "w") as fw:
                csvwriter = csv.writer(fw)
                csvwriter.writerow(["tfrecord", "pred_prob", "pred_class", "target_class"])
                for info_i in pred_info:
                    csvwriter.writerow(info_i)

        if args.gpu ==0:
            print('Eval Epoch %d %s Accuracy is %.2f%% (%.2f%%)'%(int(epoch),\
                dataset_name, acc_tensors[0], acc_tensors[1]))
            tb_logger.log_value('{} top1 Acc'.format(dataset_name),\
                acc_tensors[0], epoch)
            tb_logger.log_value('{} top5 Acc'.format(dataset_name),\
                acc_tensors[1], epoch)

    return acc_tensors[0].item(), acc_tensors[1].item()


def test_nuswide(model, test_loader, args, epoch, tb_logger,\
    dataset_name="NUSWide", save_feat=False):
    ### test nuswide dataset
    with torch.no_grad():
        print('==> Evaluation...')     
        model.eval()

        predictions = []
        predictions_prob_all = []
        targets = []
        tfrecord_names_all = []

        batch_idx = 0
        for batch in tqdm(test_loader):
            if save_feat:
                ## concatenate all features
                tfrecord_names_batch = []
                features_batch = []

            ## outputs, feat, target, feat_reconstruct
            tfrecord_names = batch[3]
            outputs, target, _, q = model(batch, args, is_eval=True)
            _, pred = torch.sigmoid(outputs).topk(3, 1, True, True)
            predictions_prob_all.append(torch.sigmoid(outputs))
            predictions.append(pred)
            targets.append(target)
            for tfrecord_name in tfrecord_names:
                tfrecord_names_all.append(str(tfrecord_name))
                    
            if save_feat:
                assert(len(tfrecord_names) == len(q))
                for tfrecord_name, feature in zip(tfrecord_names, q):
                    tfrecord_names_batch.append(str(tfrecord_name))
                    features_batch.append(feature.detach().cpu().numpy().tolist())
                    
                if args.gpu == 0:
                    save_names_path = os.path.join(args.exp_dir, "{}_tfrecord_names_all_{}.txt".format(dataset_name, batch_idx))
                    save_feat_path = os.path.join(args.exp_dir, "{}_save_feats_{}.npy".format(dataset_name, batch_idx))
                    np.save(save_feat_path, np.array(features_batch))
                    with open(save_names_path, "w") as fw:
                        for tfrecord_name in tfrecord_names_batch:
                            fw.write(tfrecord_name + "\n")
            batch_idx += 1

        predictions = torch.cat(predictions, dim=0).cuda(args.gpu)
        predictions_prob_all = torch.cat(predictions_prob_all, dim=0).cuda(args.gpu)
        targets = torch.cat(targets, dim=0).cuda(args.gpu)
        ## collect from all gpus
        predictions = concat_all_gather(predictions)
        predictions_prob_all = concat_all_gather(predictions_prob_all)
        targets = concat_all_gather(targets)
        ## from tensor to array
        predictions_npy = predictions.detach().cpu().numpy()
        predictions_prob_all_npy = predictions_prob_all.detach().cpu().numpy()
        targets_npy = targets.detach().cpu().numpy()

        c_precision, c_recall, c_f1,\
            o_precision, o_recall, o_f1, mAP = precision_recall_map(predictions_npy,\
                predictions_prob_all_npy, targets_npy, args.num_class)

        if args.gpu ==0:
            print('Eval Epoch %d %s C-F1 is %.2f%% O-F1 is %.2f%% mAP is %.2f%%'%(int(epoch),\
                dataset_name, c_f1, o_f1, mAP))
            tb_logger.log_value('{} C-Precision'.format(dataset_name),\
                c_precision, epoch)
            tb_logger.log_value('{} C-Recall'.format(dataset_name),\
                c_recall, epoch)
            tb_logger.log_value('{} C-F1'.format(dataset_name),\
                c_f1, epoch)
            tb_logger.log_value('{} O-Precision'.format(dataset_name),\
                o_precision, epoch)
            tb_logger.log_value('{} O-Recall'.format(dataset_name),\
                o_recall, epoch)
            tb_logger.log_value('{} O-F1'.format(dataset_name),\
                o_f1, epoch)
            tb_logger.log_value('{} mAP'.format(dataset_name),\
                mAP, epoch)

        if save_feat:
            assert(len(tfrecord_names_all) == len(predictions_npy))
            mapping_tfrecord2idx = {}
            for idx, tfrecord_name in enumerate(tfrecord_names_all):
                mapping_tfrecord2idx[tfrecord_name] = int(idx)
            save_pred_path = os.path.join(args.exp_dir, "{}_pred.npy".format(dataset_name))
            save_pred_prob_all_path = os.path.join(args.exp_dir, "{}_pred_prob_all.npy".format(dataset_name))
            save_target_path = os.path.join(args.exp_dir, "{}_gt.npy".format(dataset_name))
            np.save(save_pred_path, predictions_npy)
            np.save(save_pred_prob_all_path, predictions_prob_all_npy)
            np.save(save_target_path, targets_npy)
            save_mapping_path = os.path.join(args.exp_dir, "{}_mapping_tfrecord2idx.json".format(dataset_name))
            with open(save_mapping_path, "w") as fw:
                json.dump(mapping_tfrecord2idx, fw)

    return c_f1, o_f1, mAP


def save_checkpoint(state, is_best, filename='checkpoint_latest.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint_best.pth')
    return


def save_checkpoint_nuswide(epoch, arch, model, optimizer,\
    cf1_max, of1_max, mAP_max, epoch_best_cf1,\
        epoch_best_of1, epoch_best_mAP, filename):
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_cf1': cf1_max,
        'best_of1': of1_max,
        'best_mAP': mAP_max,
        'epoch_best_cf1': epoch_best_cf1,
        'epoch_best_of1': epoch_best_of1,
        'epoch_best_mAP': epoch_best_mAP,
    }, is_best=False, filename=filename)
    return


def save_checkpoint_webvision(epoch, arch, model, optimizer,\
    acc1_web, acc5_web, acc1_imgnet, acc5_imgnet,\
        epoch_best_web, epoch_best_imgnet, filename):
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc_web': [acc1_web, acc5_web],
        'best_acc_imgnet': [acc1_imgnet, acc5_imgnet],
        'epoch_best_web': epoch_best_web,
        'epoch_best_imgnet': epoch_best_imgnet,
    }, is_best=False, filename=filename)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-3)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)
        return

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = (target.size(0) + 1e-7)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k, :].sum(0)
            correct_k = correct_k.view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_multi_label(output, target, topk=-1, target_valid=None):
    """Computes the accuracy over all classes with binary prediction after sigmoid"""
    with torch.no_grad():
        if topk > 0:
            ## 只统计前k大的样本是否正确
            output_topk, _ = output.topk(topk, 1, True, True)
            output_topk_min, _ = output_topk.min(dim=1, keepdim=True)
            output_bin = (output >= output_topk_min).view(-1)

        else:
            ## 依赖阈值
            output_bin = (output >= 0.5).view(-1)
        target_bin = (target >= 0.5).view(-1)
        if not (target_valid is None):
            target_valid = target_valid.view(-1)
            output_bin = output_bin[target_valid]
            target_bin = target_bin[target_valid]
        correct = output_bin.eq(target_bin)
        return correct.float().sum() * 100/len(output_bin)


if __name__ == '__main__':
    main()



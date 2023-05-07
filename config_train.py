import argparse

parser = argparse.ArgumentParser(description='PyTorch WebVision/NUSWide Training')
#### Root Path And Pathlist
parser.add_argument('--root_dir', default='../dataset/',
                    help='path to dataset folder')
parser.add_argument('--pathlist_web', default='',
                    help='path to the imglist for webvision web data')
parser.add_argument('--root_dir_test_web', default='',
                    help='path to dataset folder for webvision web data')
parser.add_argument('--pathlist_test_web', default='',
                    help='path to the imglist for webvision web test data')
parser.add_argument('--root_dir_test_target', default='',
                    help='path to dataset folder for webvision imgnet test data')
parser.add_argument('--pathlist_test_target', default='',
                    help='path to the imglist for webvision imgnet test data')
parser.add_argument('--root_dir_t', default='',
                    help='path to dataset folder for fewshot target domain data')
parser.add_argument('--pathlist_t', default='',
                    help='path to the imglist for fewshot target domain data')
parser.add_argument('--exp-dir', default='experiment/MoPro_V1', type=str,
                    help='experiment directory')

#### Dataset choice/preparation
parser.add_argument('--nuswide', action='store_true', default=False,
                    help='use nuswide dataset if True else WebVision')
parser.add_argument('--topk', type=int, default=50,
                    help='TOP K fewshots')
parser.add_argument('--use_fewshot', action='store_true', default=False,
                    help='use fewshot dataset for training')  
parser.add_argument('--use_soft_label', type=int, default=7,
                    help='use soft-label for \
                    training the global classifier\
                    softlabel=0(none); softlabel=1(bootstrapping)\
                    softlabel=2(label smoothing); softlabel=3(SCC)\
                    softlabel=4(SCC+); softlabel=5(NCR);\
                    softlabel=6(memorybank), softlabel=7(collective bootstrapping)')
parser.add_argument('--beta', default=0.1, type=float,
                    help='weight to combine soft and hard\
                    target classification\
                    (including weight for collective boostrapping loss)')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='use mix-up for classifier finetune')

#### Specific Configurations
parser.add_argument('--use_meta_weights', default=0, type=int,
                    help='use meta information as sample weights,\
                         -1=none; 0=default; 1=single')
parser.add_argument('--pseudo_fewshot', action='store_true', default=False,
                    help='use pseudo fewshot info with highly confident meta info')

#### Basic/General Configurations
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='only for evaluation')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=['resnet50','resnet50x2','resnet50x4',\
                             'resnetD50','vgg','alexnet','bcnn'])
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--init-proto-epoch', default=10, type=int, 
                    help='epoch to update the init-proto-type')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[40, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--num-class', default=1000, type=int)
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--update_proto_freq', default=1, type=int,
                    help="update prototype frequency epoch frequency; 0 means do not update")
parser.add_argument('--warmup_epoch', default=5, type=int,
                    help='warm up epoch for learning rate update')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.999, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--temperature', default=0.1, type=float,
                    help='contrastive temperature')
parser.add_argument('--save_feat', action='store_true', default=False,
                    help='save features of each sample')
parser.add_argument('--dry_run', action='store_true', default=False,
                    help='use small number of samples for debug')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='use imagenet pretrained backbone')
parser.add_argument('--w-inst', default=1, type=float,
                    help='weight for instance contrastive loss')
parser.add_argument('--w-recn', default=1, type=float,
                    help='weight for projection-reconstruction loss')
parser.add_argument('--w-proto', default=1, type=float,
                    help='weight for prototype contrastive loss')
parser.add_argument('--start_clean_epoch', default=10, type=int,
                    help='epoch to start noise cleaning')
parser.add_argument('--frozen_encoder_epoch', default=5, type=int,
                    help='froze encoder and only train classifier')
parser.add_argument('--pseudo_th', default=0.6, type=float,
                    help='threshold for correcting pseudo labels')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='weight to combine model prediction and prototype prediction')
parser.add_argument('--annotation', default='./pseudo_label.json',
                    help='path to pseudo-label annotation')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='finetune encoder together with classifier')
parser.add_argument('--fast_eval', default=0, type=int,
                    help='fast evaluation for webvision = 1 or google500 = 2')
parser.add_argument('--rebalance', action='store_true', default=False,
                    help='rebalance samples for classifier retraining')
parser.add_argument('--no_color_transform', action='store_true', default=False,
                    help='do not perform color transform on images')

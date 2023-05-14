import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('.')
from backbone.basenet import AlexNet_Encoder, VGG_Encoder, BCNN_encoder
from backbone.resnet import resnet50
from backbone.resnetd import resnetD50
from backbone.classifier import Normalize, MLP_classifier
import math


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def JSD(p, q, reduction="batchmean"):
    log_mean_output = (0.5 * (p + q)).log()
    return 0.5 * (F.kl_div(log_mean_output, p, reduction=reduction) +\
        F.kl_div(log_mean_output, q, reduction=reduction))


class CAPro(nn.Module):

    def __init__(self, args):
        super(CAPro, self).__init__()
        ##=========================================================================##
        ## 设置特征抽取器
        ##=========================================================================##
        if args.arch == 'resnet50':
            ### this is the default
            base_encoder_q = resnet50(pretrained=args.pretrained, width=1)
            base_encoder_k = resnet50(pretrained=args.pretrained, width=1)
        elif args.arch == 'resnet50x2':
            base_encoder_q = resnet50(pretrained=args.pretrained, width=2)
            base_encoder_k = resnet50(pretrained=args.pretrained, width=2)
        elif args.arch == 'resnet50x4':
            base_encoder_q = resnet50(pretrained=args.pretrained, width=4)
            base_encoder_k = resnet50(pretrained=args.pretrained, width=4)
        elif args.arch == 'resnetD50':
            base_encoder_q = resnetD50(pretrained=args.pretrained)
            base_encoder_k = resnetD50(pretrained=args.pretrained)
        elif args.arch == 'vgg':
            ## 默认num_out_channel=4096
            base_encoder_q = VGG_Encoder(pretrained=args.pretrained)
            base_encoder_k = VGG_Encoder(pretrained=args.pretrained)
        elif args.arch == 'bcnn':
            ## 默认num_out_channel=512**2
            base_encoder_q = BCNN_encoder(pretrained=args.pretrained, num_out_channel=512**2)
            base_encoder_k = BCNN_encoder(pretrained=args.pretrained, num_out_channel=512**2)
        elif args.arch == 'alexnet':
            ## 默认num_out_channel=4096
            base_encoder_q = AlexNet_Encoder(pretrained=args.pretrained)
            base_encoder_k = AlexNet_Encoder(pretrained=args.pretrained)
        else:
            raise NotImplementedError('model not supported {}'.format(args.arch))
        ## encoder
        self.encoder_q = base_encoder_q
        ## momentum encoder
        self.encoder_k = base_encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        ##=========================================================================##
        ## 设置高维向量=>低维向量的投影器
        ## 如果args.low_dim == -1则不设置projection直接用分类embedding
        ##=========================================================================##
        if args.nuswide:
            ## 对于多分类问题(e.g., NUSWide)每个类都具备单独的特征投影器
            if args.low_dim != -1:
                self.low_dim = args.low_dim
                self.projection = nn.Sequential(*[
                        nn.Linear(self.encoder_q.num_out_channel, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, self.low_dim),
                        Normalize(2),
                    ])
                self.low_dim_class = self.low_dim
                self.projection_class = nn.ModuleList([nn.Sequential(*[
                    nn.Linear(self.low_dim, self.low_dim_class),
                    Normalize(2),
                    ]) for _ in range(args.num_class)])
                self.projection_back = nn.Sequential(*[
                    nn.Linear(self.low_dim, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, self.encoder_q.num_out_channel),
                ])
            else:
                self.low_dim = self.encoder_q.num_out_channel
                self.low_dim_class = self.low_dim
                self.projection = nn.Sequential(*[
                    nn.Identity(),
                    Normalize(2),
                ])
                self.projection_class = nn.ModuleList([nn.Sequential(*[
                    nn.Identity(),
                    Normalize(2),
                ]) for _ in range(args.num_class)])
                self.projection_back = nn.Sequential(*[
                    nn.Identity(),
                ])
        else:
            ## 对于单分类问题(e.g., ImageNet)各个类别间共享相同的特征投影器
            if args.low_dim != -1:
                self.low_dim = args.low_dim
                self.low_dim_class = self.low_dim
                if args.arch == 'bcnn':
                    self.projection = nn.Sequential(*[
                        nn.Linear(self.encoder_q.num_out_channel, self.low_dim),
                        Normalize(2),
                    ])
                    self.projection_back = nn.Sequential(*[
                        nn.Linear(self.low_dim, self.encoder_q.num_out_channel),
                    ])
                else:
                    self.projection = nn.Sequential(*[
                        nn.Linear(self.encoder_q.num_out_channel, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, self.low_dim),
                        Normalize(2),
                    ])
                    self.projection_back = nn.Sequential(*[
                        nn.Linear(self.low_dim, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, self.encoder_q.num_out_channel),
                    ])
            else:
                self.low_dim = self.encoder_q.num_out_channel
                self.low_dim_class = self.low_dim
                self.projection = nn.Sequential(*[
                    nn.Identity(),
                    Normalize(2)
                ])
                self.projection_back = nn.Sequential(*[
                    nn.Identity(),
                ])
        ##=========================================================================##
        ## 设置关系抽取器
        ## 如果使用参数化的relation module来自己学习距离关系则去掉以下注释
        ## 避免过多参数量可以用l2 norm + cosine similarity代表relation score
        ## 但直接用cosine similarity可能并不是最优解
        ##=========================================================================##
        self.relation = MLP_classifier(num_class=1, in_channel=self.low_dim * 2,\
            num_hidden=128, use_norm=False, use_sigmoid=True)
        self.relation.apply(init_weights)
        ##=========================================================================##
        ## 设置分类器 不使用norm
        ##=========================================================================##
        self.classifier = MLP_classifier(num_class=args.num_class,\
            in_channel=self.encoder_q.num_out_channel, use_norm=False)
        self.classifier.apply(init_weights)
        ##=========================================================================##
        ## 设置low-dim分类器 使用norm
        ##=========================================================================##
        if args.nuswide:
            ## 相应地重建模块也需要逐类别设置
            self.classifier_projection = nn.ModuleList([MLP_classifier(num_class=1,\
                in_channel=self.low_dim_class, use_norm=True) for _ in range(args.num_class)])
            for mlp_classifier in self.classifier_projection:
                mlp_classifier.apply(init_weights)
        else:
            ## 各类别共享一个重建模块
            self.classifier_projection = MLP_classifier(num_class=args.num_class,\
                in_channel=self.low_dim_class, use_norm=True)
            self.classifier_projection.apply(init_weights)
        ##=========================================================================##
        ## 设置自监督对比学习用的已访问队列
        ##=========================================================================##
        self.register_buffer("queue", torch.randn(self.low_dim, args.moco_queue))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        ##=========================================================================##
        ## 设置每个类别的prototype, 访问次数, 访问距离
        ##=========================================================================##
        if args.nuswide:
            self.register_buffer("prototypes", torch.zeros((args.num_class, 2, self.low_dim_class)))
            self.register_buffer("prototypes_visited", torch.zeros(args.num_class, 2))
            self.register_buffer("prototypes_density", torch.ones(args.num_class, 2)*args.temperature)
            self.register_buffer("prototypes_distance", torch.zeros(args.num_class, 2))
        else:
            self.register_buffer("prototypes", torch.zeros((args.num_class, self.low_dim_class)))
            self.register_buffer("prototypes_visited", torch.zeros((args.num_class,)))
            self.register_buffer("prototypes_density", torch.ones((args.num_class,))*args.temperature)
            self.register_buffer("prototypes_distance", torch.zeros((args.num_class,)))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
        return
    
    @torch.no_grad()
    def _initialize_prototype_features(self):
        """
        initialize prototype features by average
        """
        ## average over all
        self.prototypes = self.prototypes / (torch.unsqueeze(self.prototypes_visited, dim=-1) + 1e-6)
        ## normalize
        self.prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        self._print_prototype_features()
        ## 重新对prototype visited数目清零
        self.prototypes_visited *= 0
        return
    
    @torch.no_grad()
    def _zero_prototype_features(self):
        self.prototypes *= 0.
        self.prototypes_distance *= 0.
        self.prototypes_visited *= 0.
        return
    
    @torch.no_grad()
    def _print_prototype_features(self):
        print("prototype visited total", int(torch.sum(self.prototypes_visited).item()))
        prototype_nonzero = self.prototypes[self.prototypes_visited>0]
        if len(self.prototypes.size()) == 3:
            prototype_nonzero = self.prototypes[self.prototypes_visited[...,1]>0]
        elif len(self.prototypes.size()) == 2:
            prototype_nonzero = self.prototypes[self.prototypes_visited>0]
        else:
            raise ValueError("Invalid prototype size")
        print("prototype features", prototype_nonzero.size(), prototype_nonzero)
        return

    @torch.no_grad()
    def _update_prototype_density(self):
        """
        update prototype density for temperature
        """
        ## update density density
        for class_id in range(self.prototypes_visited.size(0)):
            if self.nuswide:
                for proto_id in range(self.prototypes_visited.size(1)):
                    num_visited = self.prototypes_visited[class_id][proto_id]
                    distance_sum = self.prototypes_distance[class_id][proto_id]
                    self.prototypes_density[class_id][proto_id] = distance_sum / (num_visited*torch.log(num_visited+10.)+1e-7)
            else:
                num_visited = self.prototypes_visited[class_id]
                distance_sum = self.prototypes_distance[class_id]
                self.prototypes_density[class_id] = distance_sum / (num_visited*torch.log(num_visited+10.)+1e-7)                
        self.prototypes_distance *= 0
        self.prototypes_visited *= 0
        return
    
    @torch.no_grad()
    def _print_norm_tensor(self, input_tensor, tensor_name="tensor", power=2):
        """
        print the l-N norm of input tensor
        """
        norm_tensor = input_tensor.pow(power).sum(1, keepdim=True).pow(1./power)
        print("L{} norm of {} = {}".format(power, tensor_name, norm_tensor))
        return

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, args):
        ## keys: features commonly shared for all classes
        ## gather keys before updating queue
        ptr = int(self.queue_ptr)
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        assert args.moco_queue % batch_size == 0  # for simplicity
        if ptr + batch_size > self.queue.size(1):
            ptr = self.queue.size(1) - batch_size
        ## replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ## L2 normalization on the features
        self.queue = F.normalize(self.queue, dim=0)
        ## move pointer
        ptr = (ptr + batch_size) % args.moco_queue  
        ## update the pointer
        self.queue_ptr[0] = ptr
        return

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # get batchsize from current gpu
        batch_size_this = x.shape[0]
        # gather from all gpus
        x_gather = concat_all_gather(x)
        # get batchsize from all gpus
        batch_size_all = x_gather.shape[0]
        # get the number of gpus
        num_gpus = batch_size_all // batch_size_this
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # x (original order) -> x[idx_shuffle] (shuffled order)
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        # index for restoring
        # idx_shuffle -> argsort from 0~batch_size_all-1 -> recover by idx_unshuffle
        idx_unshuffle = torch.argsort(idx_shuffle)
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        # get re-ordered x by id_shuffle on this gpu
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # get batchsize from current gpu
        batch_size_this = x.shape[0]
        # gather from all gpus
        x_gather = concat_all_gather(x)
        # get batchsize from all gpus
        batch_size_all = x_gather.shape[0]
        # get the number of gpus
        num_gpus = batch_size_all // batch_size_this
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        # get the local batch x from restored order
        return x_gather[idx_this]

    @torch.no_grad()
    def _accumulate_prototype_features(self, img, target, args, is_eval):
        ## 累加特征需要按照NUSWIDE/WEBVISION数据集进行分类讨论
        if args.nuswide:
            ## 如果是验证集来可视化特征仅仅需要当前batch的特征即可
            if is_eval:
                k_compress = self.projection(self.encoder_k(img))
                k_compress_class = [projection_class_i(k_compress) for projection_class_i in self.projection_class]
                features = k_compress_class
                targets = target
            else:
                img, idx_unshuffle = self._batch_shuffle_ddp(img)
                k_compress = self.projection(self.encoder_k(img))
                k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
                k_compress_class = [projection_class_i(k_compress) for projection_class_i in self.projection_class]
                features = [concat_all_gather(k_compress_class_i) for k_compress_class_i in k_compress_class]
                targets = concat_all_gather(target)
            for sample_idx, label in enumerate(targets):
                ## 多分类任务应当对所有类别进行累加
                for label_idx, label_val in enumerate(label):
                    if int(label_val) < 0:
                        continue
                    self.prototypes[int(label_idx)][int(label_val)] += features[int(label_idx)][sample_idx]
                    self.prototypes_visited[int(label_idx)][int(label_val)] += 1
        else:
            ## 如果是验证集来可视化特征仅仅需要当前batch的特征即可
            if is_eval:
                k_compress = self.projection(self.encoder_k(img))
                features = k_compress
                targets = target
            else:
                img, idx_unshuffle = self._batch_shuffle_ddp(img)
                k_compress = self.projection(self.encoder_k(img))
                k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
                features = concat_all_gather(k_compress)
                targets = concat_all_gather(target)
            for feat, label in zip(features, targets):
                ## 单分类任务只需要累加对应的类即可
                self.prototypes[int(label)] += feat
                self.prototypes_visited[int(label)] += 1
        return

    @torch.no_grad()
    def _update_prototype_features(self, features, targets, args):
        if args.nuswide:
            for sample_idx, label in enumerate(targets):
                for label_idx, label_val in enumerate(label):
                    if int(label_val) < 0:
                        continue
                    self.prototypes[int(label_idx)][int(label_val)] =\
                        self.prototypes[int(label_idx)][int(label_val)]*args.proto_m +\
                        (1-args.proto_m)*features[int(label_idx)][sample_idx]
        else:
            for feat, label in zip(features, targets):
                self.prototypes[int(label)] = self.prototypes[int(label)]*args.proto_m + (1-args.proto_m)*feat
        # normalize
        self.prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        return

    def forward_nuswide(self, batch, args, is_eval, is_proto_init, is_proto,\
                        is_clean, is_update_proto=True):
        """前向传播for train and eval
        batch: 当前输入,包括图像;标签;domain标签(web or fewshot);(输入强数据增强图像);图像pathlist
        args: 全局训练参数设置
        is_eval: 是否验证集推理, 如果是则仅仅返回分类结果&标签&重建的高维特征&高维特征
        is_proto_init: 是否初始化prototype; True 累加特征;
        is_proto: 是否进行prototype更新, 如果不更新则不会进行对比学习&噪声过滤&标签修正
        is_clean: 是否进行标签修正&噪声过滤
        """
        img = batch[0].cuda(args.gpu, non_blocking=True)
        target = batch[1].cuda(args.gpu, non_blocking=True)
        ##=========================================================================##
        ## 初始化/更新模型prototype
        ##=========================================================================##
        if is_proto_init:
            ## 初始化prototype特征=>按类累加
            self._accumulate_prototype_features(img, target, args, is_eval)
            return
        ##=========================================================================##
        ## 特征抽取器提取特征; 压缩部分得到低维特征表示; 分类器进行分类
        ##=========================================================================##
        q = self.encoder_q(img)
        output = self.classifier(q)
        ## nuswide数据集下的压缩特征为列表
        q_compress = self.projection(q)
        q_compress_class = [projection_class_i(q_compress) for projection_class_i in self.projection_class]
        ##=========================================================================##
        ## 特征压缩与重建
        ##=========================================================================##
        if is_proto:
            ## 2) 训练后期仅仅更新重建部分 仅用于防止pytorch报错(存在未更新参数)
            q_reconstruct = self.projection_back(q_compress.detach().clone())
        else:
            ## 1) 训练初期同时更新压缩和重建部分来训练projector生成合理的low-dim特征
            q_reconstruct = self.projection_back(q_compress)
        ##=========================================================================##
        ## 测试推理仅返回预测结果
        ##=========================================================================##
        if is_eval:
            return output, target, q_compress, q
        ##=========================================================================##
        ## 动量更新特征提取器(除了仅finetune relation module时不更新)
        ##=========================================================================##
        with torch.no_grad():  # no gradient
            self._momentum_update_key_encoder(args)
        ##=========================================================================##
        ## 来自FewShot(pseudo) domain的样本大概率是干净样本
        ##=========================================================================##
        if is_eval:
            domain = batch[2].cuda(args.gpu, non_blocking=True)
        else:
            domain = batch[3].cuda(args.gpu, non_blocking=True)
        domain = domain.view(-1, args.num_class)
        fewshot_idx = (domain != 0)
        ##=========================================================================##
        ## 获得每个样本的权重以及目标meta_soft_target标签
        ##=========================================================================##
        img_aug = batch[2].cuda(args.gpu, non_blocking=True)
        sample_weight = batch[6].cuda(args.gpu, non_blocking=True)
        target_meta_soft = batch[7].cuda(args.gpu, non_blocking=True)
        ##=========================================================================##
        ## 训练投影后空间的分类器使投影后的特征具备可区分性
        ##=========================================================================##
        ## 计算每个样本经过投影模块后再经过多分类的prototype提纯模块=>每个类一个单独的投影模块
        output_compress = torch.cat([classifier_projection_i(q_compress_class_i) for q_compress_class_i,\
            classifier_projection_i in zip(q_compress_class, self.classifier_projection)], dim=-1)
        ##=========================================================================##
        ## instance-instance对比学习
        ##=========================================================================##
        with torch.no_grad():  # no gradient
            ## shuffle for making use of BN undo shuffle
            img_aug, idx_unshuffle = self._batch_shuffle_ddp(img_aug)
            k_compress = self.projection(self.encoder_k(img_aug))
            k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
        ## compute instance logits
        l_pos = torch.einsum('nc,nc->n', [q_compress, k_compress]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_compress, self.queue.detach().clone()])
        inst_logits = torch.cat([l_pos, l_neg], dim=1)/args.temperature
        inst_labels = torch.zeros(inst_logits.shape[0], dtype=torch.long).cuda()
        ##=========================================================================##
        ## 访问过的队列 入库 & 出库
        ##=========================================================================##
        self._dequeue_and_enqueue(k_compress, args)
        ##=========================================================================##
        ## 得到更新后的queue的分类结果
        ##=========================================================================##
        if args.use_soft_label in [7, 9]:
            with torch.no_grad():  # no gradient
                queue_sim = torch.einsum('nc,ck->nk', [q_compress, self.queue.detach().clone()])
                queue_sim = torch.clamp(queue_sim, min=0, max=1)+1e-7
                queue_sim = queue_sim/torch.sum(queue_sim, dim=1, keepdim=True)
                queue_t_class = [projection_class_i(self.queue.detach().clone().t()) for projection_class_i in self.projection_class]
                output_queue_t = torch.cat([classifier_projection_i(queue_t_class_i) for queue_t_class_i,\
                    classifier_projection_i in zip(queue_t_class, self.classifier_projection)], dim=-1)
                output_queue_t = torch.sigmoid(output_queue_t)
                # print("output_queue_t", output_queue_t.size(), (output_queue_t>0.5)[0])
                prototypes_queue = self.prototypes.clone().detach()
                logits_proto_queue_t = torch.cat([torch.unsqueeze(torch.mm(queue_t_class_i, prototype.t()), 1) for queue_t_class_i,\
                    prototype in zip(queue_t_class, prototypes_queue)], dim=1)/args.temperature
                logits_proto_queue_t = F.softmax(logits_proto_queue_t, dim=2)[...,1]
                # print("logits_proto_queue_t", logits_proto_queue_t.size(), (logits_proto_queue_t>0.5)[0])
                soft_label_queue_t = args.alpha*output_queue_t + (1-args.alpha)*logits_proto_queue_t
                # print("soft_label_queue_t", soft_label_queue_t.size(), (soft_label_queue_t>0.5)[0])
                target_queue = torch.einsum('nc,ck->nk', [queue_sim, soft_label_queue_t])
                target_queue = (torch.clamp(target_queue, min=0, max=1)).detach().clone()
        else:
            target_queue = None
        ##=========================================================================##
        ## instance-prototype对比学习
        ##=========================================================================##
        if is_proto:
            # 训练后期计算protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_proto = torch.cat([torch.unsqueeze(torch.mm(q_compress_class_i, prototype.t()), 1) for q_compress_class_i,\
                prototype in zip(q_compress_class, prototypes)], dim=1)/args.temperature
        else:
            ## 训练初期不加入prototype进行对比学习
            logits_proto = 0
        ##=========================================================================##
        ## 标签修正
        ##=========================================================================##
        if is_clean:
            ## noise cleaning
            soft_label_pos = args.alpha*torch.sigmoid(output) + (1-args.alpha)*F.softmax(logits_proto, dim=2)[...,1]
            soft_label_neg = 1.-soft_label_pos
            ## keep ground truth label (OOD removal)
            clean_idx_pos = (soft_label_pos>=1/args.num_class) & (target==1)
            clean_idx_neg = (soft_label_neg>=1/args.num_class) & (target==0)
            clean_idx_pos_neg = (clean_idx_pos | clean_idx_neg)
            ## assign a new pseudo label
            ## 标签为负样本但预测值>pseudo-th=>修正为正样本
            correct_idx_pos = (soft_label_pos>=args.pseudo_th) & (target==0)
            ## 标签为正样本但预测值<1.-pseudo-th=>修正为负样本
            correct_idx_neg = (soft_label_neg>=args.pseudo_th) & (target==1)
            correct_idx_pos_neg = (correct_idx_pos | correct_idx_neg)
            if args.pseudo_fewshot:
                ## 保留fewshot样本为干净样本
                clean_idx_pos = clean_idx_pos | fewshot_idx
                clean_idx_neg = clean_idx_neg | fewshot_idx
                clean_idx_pos_neg = clean_idx_pos_neg | fewshot_idx
                ## 保证fewshot样本不被改变
                correct_idx_pos = correct_idx_pos & (~fewshot_idx)
                correct_idx_neg = correct_idx_neg & (~fewshot_idx)
                correct_idx_pos_neg = correct_idx_pos_neg & (~fewshot_idx)
            target[correct_idx_pos] = 1
            target[correct_idx_neg] = 0
            ## confident sample index
            clean_idx, _ = clean_idx_pos_neg.max(1)
            correct_idx, _ = correct_idx_pos_neg.max(1)
            clean_idx = clean_idx | correct_idx
            clean_idx_pos_neg = clean_idx_pos_neg | correct_idx_pos_neg
            target[~clean_idx_pos_neg] = -1
            clean_idx_all = concat_all_gather(clean_idx.long())
        ##=========================================================================##
        ## 筛选干净样本并更新类别原型
        ##=========================================================================##
        ## aggregate features and (pseudo) labels across all gpus        
        features = [concat_all_gather(q_compress_class_i) for q_compress_class_i in q_compress_class]
        targets = concat_all_gather(target)
        if is_clean and is_proto:
            clean_idx_all = clean_idx_all.bool()
            # update momentum prototypes with pseudo-labels
            features = [feature[clean_idx_all] for feature in features]
            targets = targets[clean_idx_all]
            if is_update_proto:
                self._update_prototype_features(features, targets, args)           
            # select only the confident samples to return
            q_compress_class = [q_compress_class_i[clean_idx] for q_compress_class_i in q_compress_class]
            q = q[clean_idx]
            q_reconstruct = q_reconstruct[clean_idx]
            output_compress = output_compress[clean_idx]
            img = img[clean_idx]
            target = target[clean_idx]
            output = output[clean_idx]
            if not (target_queue is None):
                target_queue = target_queue[clean_idx]
            sample_weight = sample_weight[clean_idx]
            logits_proto = logits_proto[clean_idx]
            target_meta_soft = target_meta_soft[clean_idx]
        elif is_proto:
            # update momentum prototypes with original labels
            if is_update_proto:
                self._update_prototype_features(features, targets, args)
        ##=========================================================================##
        ## MIX-UP应该仅仅作用于分类器学习
        ##=========================================================================##  
        if args.mixup:
            rand_index = torch.randperm(img.size()[0]).cuda(args.gpu, non_blocking=True)
            lam_mixup = np.random.beta(0.8, 0.8)
            img_mixup = lam_mixup*img + (1.0-lam_mixup)*img[rand_index, :]
            output_mixup = self.classifier(self.encoder_q(img_mixup))
            target_mixup = target[rand_index]
            triplet_mixup = [lam_mixup, output_mixup, target_mixup]
        else:
            triplet_mixup = None

        return output, target, q_reconstruct, q, q_compress_class,\
            sample_weight, target_meta_soft, output_compress,\
                inst_logits, inst_labels, logits_proto,\
                    triplet_mixup, target_queue
     
    def forward_webvision(self, batch, args, is_eval,\
                          is_proto_init, is_proto, is_clean, is_update_proto=True):
        """前向传播for train and eval
        batch: 当前输入,包括图像;标签;domain标签(web or fewshot);(输入强数据增强图像);图像pathlist
        args: 全局训练参数设置
        is_eval: 是否验证集推理, 如果是则仅仅返回分类结果&标签&重建的高维特征&高维特征
        is_proto_init: 是否初始化prototype; True 累加特征;
        is_proto: 是否进行prototype更新, 如果不更新则不会进行对比学习&噪声过滤&标签修正
        is_clean: 是否进行标签修正&噪声过滤
        """
        img = batch[0].cuda(args.gpu, non_blocking=True)
        target = batch[1].cuda(args.gpu, non_blocking=True)
        ##=========================================================================##
        ## 初始化/更新模型prototype
        ##=========================================================================##
        if is_proto_init:
            ## 初始化prototype特征=>按类累加
            self._accumulate_prototype_features(img, target, args, is_eval)
            return
        ##=========================================================================##
        ## 特征抽取器提取特征; 压缩部分得到低维特征表示; 分类器进行分类
        ##=========================================================================##
        q = self.encoder_q(img)
        output = self.classifier(q)
        ## webvision数据集下的压缩特征为单个向量
        q_compress = self.projection(q)
        ##=========================================================================##
        ## 特征压缩与重建
        ##=========================================================================##
        if is_proto:
            ## 2) 训练后期仅仅更新重建部分 仅用于防止pytorch报错(存在未更新参数)
            q_reconstruct = self.projection_back(q_compress.detach().clone())
        else:
            ## 1) 训练初期同时更新压缩和重建部分来训练projector生成合理的low-dim特征
            q_reconstruct = self.projection_back(q_compress)
        ##=========================================================================##
        ## 测试推理仅返回预测结果
        ##=========================================================================##
        if is_eval:
            return output, target, q_compress, q
        ##=========================================================================##
        ## 动量更新特征提取器(除了仅finetune relation module时不更新)
        ##=========================================================================##
        with torch.no_grad():  # no gradient
            self._momentum_update_key_encoder(args)
        ##=========================================================================##
        ## 来自FewShot(pseudo) domain的样本大概率是干净样本
        ##=========================================================================##
        if is_eval:
            domain = batch[2].cuda(args.gpu, non_blocking=True).view(-1)
        else:
            domain = batch[3].cuda(args.gpu, non_blocking=True).view(-1)
        fewshot_idx = (domain > 0)
        ##=========================================================================##
        ## 获得每个样本的权重以及目标meta_soft_target标签
        ##=========================================================================##
        img_aug = batch[2].cuda(args.gpu, non_blocking=True)
        sample_weight = batch[6].cuda(args.gpu, non_blocking=True)
        target_meta_soft = batch[7].cuda(args.gpu, non_blocking=True)
        ##=========================================================================##
        ## 训练投影后空间的分类器使投影后的特征具备可区分性
        ##=========================================================================##
        ## 统一特征直接计算分类结果
        output_compress = self.classifier_projection(q_compress)
        ##=========================================================================##
        ## instance-instance对比学习
        ##=========================================================================##
        with torch.no_grad():  # no gradient
            ## shuffle for making use of BN undo shuffle
            img_aug, idx_unshuffle = self._batch_shuffle_ddp(img_aug)
            k_compress = self.projection(self.encoder_k(img_aug))
            k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
        ## compute instance logits
        l_pos = torch.einsum('nc,nc->n', [q_compress, k_compress]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_compress, self.queue.detach().clone()])
        inst_logits = torch.cat([l_pos, l_neg], dim=1)/args.temperature
        inst_labels = torch.zeros(inst_logits.shape[0], dtype=torch.long).cuda()
        ##=========================================================================##
        ## 访问过的队列 入库 & 出库
        ##=========================================================================##
        self._dequeue_and_enqueue(k_compress, args)
        ##=========================================================================##
        ## 得到更新后的queue的分类结果
        ##=========================================================================##
        if args.use_soft_label in [7, 9]:
            with torch.no_grad():  # no gradient
                queue_sim = torch.einsum('nc,ck->nk', [q_compress, self.queue.detach().clone()])
                queue_sim = torch.clamp(queue_sim, min=0, max=1)+1e-7
                queue_sim = queue_sim/torch.sum(queue_sim, dim=1, keepdim=True)
                output_queue_t = self.classifier_projection(self.queue.detach().clone().t())
                output_queue_t = F.softmax(output_queue_t, dim=1)
                # print("output_queue_t", output_queue_t.size(), "max", torch.max(output_queue_t, dim=1)[1])
                ## 修改为同时考虑预测结果与prototype投票
                prototypes_queue = self.prototypes.clone().detach()
                logits_proto_queue_t = torch.mm(self.queue.detach().clone().t(), prototypes_queue.t())/args.temperature
                logits_proto_queue_t = F.softmax(logits_proto_queue_t, dim=1)
                # print("logits_proto_queue_t", logits_proto_queue_t.size(), "max", torch.max(logits_proto_queue_t, dim=1)[1])
                soft_label_queue_t = args.alpha*output_queue_t + (1-args.alpha)*logits_proto_queue_t
                # print("soft_label_queue_t", soft_label_queue_t.size(), torch.max(soft_label_queue_t, dim=1)[1])
                # target_queue = torch.einsum('nc,ck->nk', [queue_sim, output_queue_t])
                target_queue = torch.einsum('nc,ck->nk', [queue_sim, soft_label_queue_t])
                target_queue = torch.clamp(target_queue, min=0, max=1)
                target_queue = (target_queue/torch.sum(target_queue, dim=1, keepdim=True)).detach().clone()
        else:
            target_queue = None
        ##=========================================================================##
        ## instance-prototype对比学习
        ##=========================================================================##
        if is_proto:
            ## 训练后期计算protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_proto = torch.mm(q_compress, prototypes.t())/args.temperature
        else:
            ## 训练初期不加入prototype进行对比学习
            logits_proto = 0
        ##=========================================================================##
        ## 标签修正
        ##=========================================================================##
        if is_clean:
            ## noise cleaning
            soft_label = args.alpha*F.softmax(output, dim=1) + (1-args.alpha)*F.softmax(logits_proto, dim=1)
            ## keep ground truth label (OOD removal)
            gt_score = soft_label[target>=0,target]
            clean_idx = (gt_score>=1/args.num_class)
            ## assign a new pseudo label (label correction)
            max_score, hard_label = soft_label.max(1)
            correct_idx = (max_score>=args.pseudo_th)
            if args.pseudo_fewshot:
                ## 保留fewshot样本为干净样本
                clean_idx = clean_idx | fewshot_idx
                ## 保证fewshot样本不被改变
                correct_idx = correct_idx & (~fewshot_idx)
            target[correct_idx] = hard_label[correct_idx]
            ## confident sample index
            clean_idx = clean_idx | correct_idx
            clean_idx_all = concat_all_gather(clean_idx.long())
        ##=========================================================================##
        ## 筛选干净样本并更新类别原型
        ##=========================================================================##
        ## aggregate features and (pseudo) labels across all gpus
        features = concat_all_gather(q_compress)
        targets = concat_all_gather(target)
        if args.use_soft_label == 5:
            cls_out_gather = concat_all_gather(output)
            sample_weight_gather = (concat_all_gather(sample_weight)).detach().clone()
            q_gather = (concat_all_gather(q)).detach().clone()
            q_gather = F.normalize(q_gather, p=2, dim=1)
            bs_local = output.size(0)
            with torch.no_grad():
                target_knn_gather = (F.softmax(cls_out_gather/2, dim=1)).detach().clone()
                ###use dot similarity KNN smoothing
                q_sim = torch.einsum('nc,ck->nk', [q_gather, q_gather.t()])
                q_sim = torch.clamp(q_sim, min=0, max=1)+1e-7
                sims, ners = q_sim.topk(k=min(10, bs_local), dim=1, largest=True)
                target_knn_gather_iter = []
                for sim, ner in zip(sims, ners):
                    target_knn_gather_i = torch.index_select(target_knn_gather, dim=0, index=ner)
                    sim = sim/(torch.sum(sim)+1e-7)
                    target_knn_gather_i = (target_knn_gather_i*sim.view(-1, 1)).sum(dim=0)
                    target_knn_gather_iter.append(target_knn_gather_i.unsqueeze(0))
                target_knn_gather = torch.cat(target_knn_gather_iter, dim=0)
                target_knn_gather = target_knn_gather/torch.sum(target_knn_gather, dim=1, keepdim=True)
            triplet_ncr = (cls_out_gather, target_knn_gather, sample_weight_gather)
        else:
            triplet_ncr = None

        if is_clean and is_proto:
            ## update momentum prototypes with pseudo-labels
            clean_idx_all = clean_idx_all.bool()
            features = features[clean_idx_all]
            targets = targets[clean_idx_all]
            if is_update_proto:
                self._update_prototype_features(features, targets, args)           
            ## select only the confident samples to return
            q_compress = q_compress[clean_idx]
            q = q[clean_idx]
            q_reconstruct = q_reconstruct[clean_idx]
            output_compress = output_compress[clean_idx]
            img = img[clean_idx]
            target = target[clean_idx]
            if not (target_queue is None):
                target_queue = target_queue[clean_idx]
            output = output[clean_idx]
            sample_weight = sample_weight[clean_idx]
            logits_proto = logits_proto[clean_idx]
            target_meta_soft = target_meta_soft[clean_idx]
        elif is_proto:
            ## update momentum prototypes with original labels
            if is_update_proto:
                self._update_prototype_features(features, targets, args)
        ##=========================================================================##
        ## MIX-UP应该仅仅作用于分类器学习
        ##=========================================================================##  
        if args.mixup:
            rand_index = torch.randperm(img.size()[0]).cuda(args.gpu, non_blocking=True)
            lam_mixup = np.random.beta(0.8, 0.8)
            img_mixup = lam_mixup*img + (1.0-lam_mixup)*img[rand_index, :]
            output_mixup = self.classifier(self.encoder_q(img_mixup))
            target_mixup = target[rand_index]
            triplet_mixup = [lam_mixup, output_mixup, target_mixup]
        else:
            triplet_mixup = None

        return output, target, q_reconstruct, q, q_compress,\
            sample_weight, target_meta_soft, output_compress,\
                inst_logits, inst_labels, logits_proto,\
                    triplet_mixup, target_queue, triplet_ncr

    def analysis_nuswide(self, batch, args, is_eval):
        img = batch[0].cuda(args.gpu, non_blocking=True)
        target = batch[1].cuda(args.gpu, non_blocking=True)
        q = self.encoder_q(img)
        output = self.classifier(q)
        q_compress = self.projection(q)
        q_reconstruct = self.projection_back(q_compress)
        q_compress_class = [projection_class_i(q_compress) for projection_class_i in self.projection_class]
        output_compress = torch.cat([classifier_projection_i(q_compress_class_i) for q_compress_class_i,\
            classifier_projection_i in zip(q_compress_class, self.classifier_projection)], dim=-1)
        ##=========================================================================##
        ## 来自FewShot(pseudo) domain的样本大概率是干净样本
        ##=========================================================================##
        if is_eval:
            domain = batch[2].cuda(args.gpu, non_blocking=True).view(-1, args.num_class)
            pathlist = batch[3]
            img_index = batch[4]
        else:
            img_aug = batch[2].cuda(args.gpu, non_blocking=True)
            domain = batch[3].cuda(args.gpu, non_blocking=True).view(-1, args.num_class)
            pathlist = batch[4]
            img_index = batch[5]
        fewshot_idx = (domain > 0)

        prototypes = self.prototypes.clone().detach()
        logits_proto = torch.cat([torch.unsqueeze(torch.mm(q_compress_class_i, prototype.t()), 1) for q_compress_class_i,\
            prototype in zip(q_compress_class, prototypes)], dim=1)/args.temperature

        target_org = target.detach().clone()
        ## noise cleaning
        self_prediction = torch.sigmoid(output)
        self_prediction_compress = torch.sigmoid(output_compress)
        prototype_similarity = F.softmax(logits_proto, dim=2)[...,1]
        soft_label_pos = args.alpha*self_prediction + (1-args.alpha)*prototype_similarity
        soft_label_neg = 1.-soft_label_pos
        ## keep ground truth label
        clean_idx_pos = (soft_label_pos>=1/args.num_class) & (target==1)
        clean_idx_neg = (soft_label_neg>=1/args.num_class) & (target==0)
        clean_idx_pos_neg = (clean_idx_pos | clean_idx_neg)
        ## assign a new pseudo label
        ## 标签为负样本但预测值>pseudo-th=>修正为正样本
        correct_idx_pos = (soft_label_pos>=args.pseudo_th) & (target==0)
        ## 标签为正样本但预测值<1.-pseudo-th=>修正为负样本
        correct_idx_neg = (soft_label_neg>=args.pseudo_th) & (target==1)
        correct_idx_pos_neg = (correct_idx_pos | correct_idx_neg)
        if args.pseudo_fewshot:
            ## 保留fewshot样本为干净样本
            clean_idx_pos = clean_idx_pos | fewshot_idx
            clean_idx_neg = clean_idx_neg | fewshot_idx
            clean_idx_pos_neg = clean_idx_pos_neg | fewshot_idx
            ## 保证fewshot样本不被改变
            correct_idx_pos = correct_idx_pos & (~fewshot_idx)
            correct_idx_neg = correct_idx_neg & (~fewshot_idx)
            correct_idx_pos_neg = correct_idx_pos_neg & (~fewshot_idx)
        target[correct_idx_pos] = 1
        target[correct_idx_neg] = 0
        ## confident sample index
        clean_idx, _ = clean_idx_pos_neg.max(1)
        correct_idx, _ = correct_idx_pos_neg.max(1)
        clean_idx = clean_idx | correct_idx
        clean_idx_pos_neg = clean_idx_pos_neg | correct_idx_pos_neg
        target[~clean_idx_pos_neg] = -1
        ## 返回原始标签; 修正后标签; 模型预测输出值; 模型预测输出值(低维); 与原型的相似度; 软标签
        return clean_idx, target_org, target, self_prediction, self_prediction_compress,\
            prototype_similarity, soft_label_pos, domain, pathlist
    
    def analysis_webvision(self, batch, args, is_eval):
        img = batch[0].cuda(args.gpu, non_blocking=True)
        target = batch[1].cuda(args.gpu, non_blocking=True)
        q = self.encoder_q(img)
        output = self.classifier(q)
        q_compress = self.projection(q)
        q_reconstruct = self.projection_back(q_compress.detach().clone())
        output_compress = self.classifier_projection(q_compress)
        ##=========================================================================##
        ## 来自FewShot(pseudo) domain的样本大概率是干净样本
        ##=========================================================================##
        if is_eval:
            domain = batch[2].cuda(args.gpu, non_blocking=True).view(-1)
            pathlist = batch[3]
            img_index = batch[4]
        else:
            img_aug = batch[2].cuda(args.gpu, non_blocking=True)
            domain = batch[3].cuda(args.gpu, non_blocking=True).view(-1)
            pathlist = batch[4]
            img_index = batch[5]
        fewshot_idx = (domain > 0)
        ## 训练后期计算protoypical logits(类别之间竞争并使用温度系数缩放)
        prototypes = self.prototypes.clone().detach()
        logits_proto = torch.mm(q_compress, prototypes.t()) / args.temperature
        target_org = target.detach().clone()
        self_prediction = F.softmax(output, dim=1)
        self_prediction_compress = F.softmax(output_compress, dim=1)
        prototype_similarity = F.softmax(logits_proto, dim=1)
        ## noise cleaning
        soft_label = args.alpha*self_prediction + (1-args.alpha)*prototype_similarity
        ## keep ground truth label
        gt_score = soft_label[target>=0,target]
        clean_idx = gt_score>=1/args.num_class
        ## assign a new pseudo label
        max_score, hard_label = soft_label.max(1)
        correct_idx = max_score>=args.pseudo_th
        target[correct_idx] = hard_label[correct_idx]
        ## confident sample index
        clean_idx = clean_idx | correct_idx
        ## 返回原始标签; 修正后标签; 模型预测输出值; 模型预测输出值(低维); 与原型的相似度; 软标签
        return clean_idx, target_org, target, self_prediction, self_prediction_compress,\
            prototype_similarity, soft_label, domain, pathlist


    def forward(self, batch, args, is_eval=False,\
        is_proto_init=False, is_proto=False, is_clean=False,\
            is_analysis=False, is_update_proto=True):
        if args.nuswide:
            if is_analysis:
                return self.analysis_nuswide(batch, args, is_eval)
            else:
                return self.forward_nuswide(batch, args, is_eval,\
                    is_proto_init, is_proto, is_clean,\
                        is_update_proto)
        else:
            if is_analysis:
                return self.analysis_webvision(batch, args, is_eval)
            else:
                return self.forward_webvision(batch, args, is_eval,\
                    is_proto_init, is_proto, is_clean,\
                        is_update_proto)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

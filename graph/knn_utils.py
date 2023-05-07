import os
import time
import torch
import numpy as np
import scipy.sparse as sp


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name, time.time() - self.start), flush=True)
        return exc_type is None

class FaissKNN():
    def __init__(self, feats, dist_def, k, knn_ofn='', build_graph_method='gpu'):
        if knn_ofn != '' and os.path.exists(knn_ofn):
                print('[faiss] read knns from {}'.format(knn_ofn))
                self.knns = [(knn[0, :].astype(np.int32), knn[1, :].astype(np.float32))
                             for knn in np.load(knn_ofn)['knns']]
        else:
            if build_graph_method == "cuda":
                import faiss as faiss_utils
                import faiss.contrib.torch_utils
                gpu_config = faiss_utils.GpuIndexFlatConfig()
                gpu_config.device = int(feats.get_device())  # The ID of the GPU
                # gpu_config.useFloat16 = True
                res = faiss_utils.StandardGpuResources()
                if dist_def == 'cosine:':
                    gpu_index_flat = faiss_utils.GpuIndexFlatIP(res, feats.size(1), gpu_config)
                else:
                    gpu_index_flat = faiss_utils.GpuIndexFlatL2(res, feats.size(1), gpu_config)
                gpu_index_flat.add(feats)
                sims, ners = gpu_index_flat.search(feats, k=k)
                self.knns = torch.cat([ners.unsqueeze(1), sims.unsqueeze(1)], dim=1)
                if knn_ofn != '':
                    os.makedirs(os.path.dirname(knn_ofn), exist_ok=True)
                    print('[faiss] save knns to {}'.format(knn_ofn))
                    np.savez_compressed(knn_ofn, knns=self.knns)
                del gpu_index_flat, res

            else:
                if torch.is_tensor(feats):
                    gpu_index = int(feats.get_device())  # The ID of the GPU
                    feats = (feats).cpu().data.numpy()  ##.half()
                else:
                    gpu_index = 0
                    feats = feats.astype('float32')
                if build_graph_method in ["gpu", "multigpu"]:
                    import faiss as faiss_utils
                    # with Timer('build index'):
                    if dist_def == 'cosine:':
                        index = faiss_utils.IndexFlatIP(feats.shape[1])
                    else:
                        index = faiss_utils.IndexFlatL2(feats.shape[1])
                    ###==================================================###
                    if build_graph_method == "multigpu":
                        #### MULTI-GPU
                        gpu_resources = []
                        ngpu = faiss_utils.get_num_gpus()
                        tempmem = -1
                        for i in range(ngpu):
                            res = faiss_utils.StandardGpuResources()
                            if tempmem >= 0:
                                res.setTempMemory(tempmem)
                            gpu_resources.append(res)

                        def make_vres_vdev(i0=0, i1=-1):
                            "return vectors of device ids and resources useful for gpu_multiple"
                            vres = faiss_utils.GpuResourcesVector()
                            vdev = faiss_utils.IntVector()
                            if i1 == -1:
                                i1 = ngpu
                            for i in range(i0, i1):
                                vdev.push_back(i)
                                vres.push_back(gpu_resources[i])
                            return vres, vdev
                        co = faiss_utils.GpuMultipleClonerOptions()
                        co.shard = True
                        gpu_vector_resources, gpu_devices_vector = make_vres_vdev(0, ngpu)
                        gpu_index_flat = faiss_utils.index_cpu_to_gpu_multiple(
                            gpu_vector_resources, gpu_devices_vector, index, co)
                    else:
                        #### SINGLE GPU
                        res = faiss_utils.StandardGpuResources()
                        gpu_index_flat = faiss_utils.index_cpu_to_gpu(res, 0, index)
                    ###==================================================###
                    gpu_index_flat.add(feats)
                    # with Timer('query topk {}'.format(k)):
                    sims, ners = gpu_index_flat.search(feats, k=k)
                    self.knns = [(np.array(ner, dtype=np.int32), np.array(sim, dtype=np.float32))
                                    for ner, sim in zip(ners, sims)]
                    if knn_ofn != '':
                        os.makedirs(os.path.dirname(knn_ofn), exist_ok=True)
                        print('[faiss] save knns to {}'.format(knn_ofn))
                        np.savez_compressed(knn_ofn, knns=self.knns)
                    del gpu_index_flat, res

                elif build_graph_method == 'approx':
                    import faiss as faiss_utils
                    res = faiss_utils.StandardGpuResources()
                    # with Timer('build index'):
                    d = feats.shape[1]
                    nlist = 256
                    m = 8
                    if dist_def == 'cosine:':
                        quantizer = faiss_utils.IndexFlatIP(d)
                    else:
                        quantizer = faiss_utils.IndexFlatL2(d)
                    index = faiss_utils.IndexIVFPQ(quantizer, d, nlist, m, 8)
                    gpu_index_flat = faiss_utils.index_cpu_to_gpu(res, 0, index)
                    assert not gpu_index_flat.is_trained
                    gpu_index_flat.train(feats)
                    assert gpu_index_flat.is_trained
                    gpu_index_flat.add(feats)
                    # with Timer('query topk {}'.format(k)):
                    sims, ners = gpu_index_flat.search(feats, k=k)
                    self.knns = [(np.array(ner, dtype=np.int32), np.array(sim, dtype=np.float32))
                                    for ner, sim in zip(ners, sims)]
                    if knn_ofn != '':
                        os.makedirs(os.path.dirname(knn_ofn), exist_ok=True)
                        print('[faiss] save knns to {}'.format(knn_ofn))
                        np.savez_compressed(knn_ofn, knns=self.knns)
                    del gpu_index_flat, res

                elif build_graph_method == 'cpu':
                    import faiss as faiss_utils
                    # with Timer('build index'):
                    if dist_def == 'cosine:':
                        index = faiss_utils.IndexFlatIP(feats.shape[1])
                    else:
                        index = faiss_utils.IndexFlatL2(feats.shape[1])
                    index.add(feats)
                    # with Timer('query topk {}'.format(k)):
                    sims, ners = index.search(feats, k=k)
                    self.knns = [(np.array(ner, dtype=np.int32), np.array(sim, dtype=np.float32))
                                    for ner, sim in zip(ners, sims)]
                    if knn_ofn != '':
                        os.makedirs(os.path.dirname(knn_ofn), exist_ok=True)
                        print('[faiss] save knns to {}'.format(knn_ofn))
                        np.savez_compressed(knn_ofn, knns=self.knns)
                    del index
                
                else:
                    raise NotImplementedError("invalid {} graph-building method".format(build_graph_method))
        return

    def get_knns(self):
        return self.knns

def global_build(feature_root, dist_def, k, save_filename="", build_graph_method='gpu'):
    if type(feature_root) is str:
        full_feat = np.load(feature_root)
    else:
        full_feat = feature_root
    if save_filename == "":
        knn_save_path = ""
    else:
        knn_save_path = save_filename + '_k{}.npz'.format(k)
    knn_graph = FaissKNN(full_feat, dist_def, k, knn_save_path, build_graph_method)
    return knn_graph
    # knns = knn_graph.get_knns()
    # del knn_graph
    # return knns

def fast_knns2spmat(knns, th_sim=0.2, edge_weight=True):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix
    eps = 1e-5
    n = len(knns)
    if torch.is_tensor(knns):
        nbrs = knns[:, 0]
        sims = knns[:, 1]
        nbrs = torch.clamp(nbrs, min=0, max=n-1)
        sims = torch.clamp(sims, min=0, max=1)
        nbrs = nbrs.cpu().data.numpy()
        sims = sims.cpu().data.numpy()
    else:
        if isinstance(knns, list):
            knns = np.array(knns)
        nbrs = knns[:, 0, :]
        sims = knns[:, 1, :]
        nbrs = np.clip(nbrs, 0, n-1)
        sims = np.clip(sims, 0, 1)
    # remove low similarity
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column (neighborhood)
    if not edge_weight:
        data = np.ones(data.shape)
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat

def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sgc_precompute(features, knn_graph, self_weight=0,\
    edge_weight=True, degree=1, tensor_ops=False):
    # if not tensor_ops:
    ## operations performed on numpy arrays
    if hasattr(knn_graph, "get_knns"):
        knns = knn_graph.get_knns()
    else:
        knns = knn_graph
    adj = fast_knns2spmat(knns, th_sim=0, edge_weight=edge_weight)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = row_normalize(adj)
    adj = adj + self_weight * sp.eye(adj.shape[0])
    adj = row_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if not torch.is_tensor(features):
        features = torch.FloatTensor(features.astype(np.float32))
    adj = adj.to(features.device)
    # with Timer('sgc_precompute', True):
    for i in range(degree):
        # features = torch.sparse.mm(adj, features)
        features = torch.spmm(adj, features)
    if torch.is_tensor(features):
        return features.detach().clone()
    else:
        return features.detach().cpu().data.numpy()





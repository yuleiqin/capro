__all__ = ['FancyPCA']


import torch


class FancyPCA:

    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec is None:
            eig_vec = torch.Tensor([
                [-0.5675, +0.7192, +0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, +0.4203],
            ]).t()
        if eig_val is None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        device = tensor.device
        if tensor.dim() == 3:
            assert tensor.size(0) == 3
            alpha = 0.1 * torch.normal(mean=torch.zeros(1, 3)).to(device)
            quatity = torch.mm(self.eig_val.to(device) * alpha, self.eig_vec.to(device))
            tensor += quatity.view(3, 1, 1)
        elif tensor.dim() == 4:
            assert tensor.size(1) == 3
            batch_size = tensor.size(0)
            alpha = 0.1 * torch.normal(mean=torch.zeros(batch_size, 3)).to(device)
            quatity = torch.mm(self.eig_val.to(device) * alpha, self.eig_vec.to(device))
            tensor += quatity.view(batch_size, 3, 1, 1)
        else:
            raise RuntimeError('FancyPCA requires input tensor to be 3D or 4D, but got {}D'.format(tensor.dim()))

        return tensor

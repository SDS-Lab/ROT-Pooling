import torch
import torch.nn as nn
from torch_geometric.utils import softmax, to_dense_batch
from pooling.rotlayers import ROT, UOT


def to_sparse_batch(x: torch.Tensor, mask: torch.Tensor = None):
    """
    Transform data with size (B, Nmax, D) to (sum_b N_b, D)
    :param x: a tensor with size (B, Nmax, D)
    :param mask: a tensor with size (B, Nmax)
    :return:
        x: with size (sum_b N_b, D)
        batch: with size (sum_b N_b,)
    """
    bs, n_max, d = x.shape
    # get num of nodes and reshape x
    num_nodes_graphs = torch.zeros_like(x[:, 0, 0], dtype=torch.int64).fill_(n_max)
    x = x.reshape(-1, d)  # total_nodes * d
    # apply mask
    if mask is not None:
        # get number nodes per graph
        num_nodes_graphs = mask.sum(dim=1)  # bs
        # mask x
        x = x[mask.reshape(-1)]  # total_nodes * d
    # set up batch
    batch = torch.repeat_interleave(input=torch.arange(bs, device=x.device), repeats=num_nodes_graphs)
    return x, batch


class ROTPooling(nn.Module):
    def __init__(self, dim: int, a0: float = None, a1: float = None, a2: float = None, a3: float = None,
                 num: int = 4, eps: float = 1e-8, f_method: str = 'badmm-e'):
        super(ROTPooling, self).__init__()
        self.dim = dim
        self.eps = eps
        self.num = num
        self.f_method = f_method

        if a0 is None:
            self.a0 = nn.Parameter(0.1 * torch.randn(1, 1), requires_grad=True)
        else:
            self.a0 = a0 * torch.ones(1, 1)

        if a1 is None:
            self.a1 = nn.Parameter(0.1 * torch.randn(1, 1), requires_grad=True)
        else:
            self.a1 = a1 * torch.ones(1, 1)

        if a2 is None:
            self.a2 = nn.Parameter(0.1 * torch.randn(1, 1), requires_grad=True)
        else:
            self.a2 = a2 * torch.ones(1, 1)

        if a3 is None:
            self.a3 = nn.Parameter(0.1 * torch.randn(1, 1), requires_grad=True)
        else:
            self.a3 = a3 * torch.ones(1, 1)

        self.linear_r1 = nn.Linear(in_features=self.dim, out_features=2 * self.dim, bias=False)
        self.linear_r2 = nn.Linear(in_features=2 * self.dim, out_features=1, bias=False)
        self.tanh = nn.Tanh()

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(dim=1)
        self.rot = ROT(num=self.num, eps=self.eps, f_method=self.f_method)

    def forward(self, x, batch):
        """
        The feed-forward function of ROT-Pooling
        :param x: (sum_b N_b, D) samples
        :param batch: (sum_b N_b, ) the index of sets in the batch
        :return:
            a pooling result with size (K, D)
        """
        q0 = softmax(self.linear_r2(self.tanh(self.linear_r1(x))), batch)  # (N, 1)
        x, mask = to_dense_batch(x, batch)  # (B, Nmax, D)
        mask = mask.unsqueeze(2)  # (B, Nmax, 1)
        q0, _ = to_dense_batch(q0, batch)  # (B, Nmax, 1)
        p0 = (torch.ones_like(x[:, 0, :]) / self.dim).unsqueeze(1)  # (B, 1, D)
        c1 = torch.bmm(x.permute(0, 2, 1), x) / x.shape[1]  # (B, D, D)
        c2 = torch.bmm(x, x.permute(0, 2, 1)) / x.shape[2]  # (B, Nmax, Nmax)
        a0 = self.softplus(self.a0)
        a1 = self.softplus(self.a1)
        a2 = self.softplus(self.a2)
        a3 = self.softplus(self.a3)
        trans = self.rot(x, c1, c2, p0, q0, a0, a1, a2, a3, mask)  # (B, Nmax, D)
        frot = self.dim * x * trans * mask  # (B, Nmax, D)
        return torch.sum(frot, dim=1, keepdim=False)  # (B, D)


class UOTPooling(nn.Module):
    def __init__(self, dim: int, a1: float = None, a2: float = None, a3: float = None,
                 num: int = 4, eps: float = 1e-8, f_method: str = 'badmm-e'):
        super(UOTPooling, self).__init__()
        self.dim = dim
        self.eps = eps
        self.num = num
        self.f_method = f_method

        if a1 is None:
            self.a1 = nn.Parameter(0.1 * torch.randn(1, 1), requires_grad=True)
        else:
            self.a1 = a1 * torch.ones(1, 1)

        if a2 is None:
            self.a2 = nn.Parameter(0.1 * torch.randn(1, 1), requires_grad=True)
        else:
            self.a2 = a2 * torch.ones(1, 1)

        if a3 is None:
            self.a3 = nn.Parameter(0.1 * torch.randn(1, 1), requires_grad=True)
        else:
            self.a3 = a3 * torch.ones(1, 1)

        self.linear_r1 = nn.Linear(in_features=self.dim, out_features=2 * self.dim, bias=False)
        self.linear_r2 = nn.Linear(in_features=2 * self.dim, out_features=1, bias=False)
        self.tanh = nn.Tanh()

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(dim=1)
        self.uot = UOT(num=self.num, eps=self.eps, f_method=self.f_method)

    def forward(self, x, batch):
        """
        The feed-forward function of ROT-Pooling
        :param x: (sum_b N_b, D) samples
        :param batch: (sum_b N_b, ) the index of sets in the batch
        :return:
            a pooling result with size (K, D)
        """
        q0 = softmax(self.linear_r2(self.tanh(self.linear_r1(x))), batch)  # (N, 1)
        x, mask = to_dense_batch(x, batch)  # (B, Nmax, D)
        mask = mask.unsqueeze(2)  # (B, Nmax, 1)
        q0, _ = to_dense_batch(q0, batch)  # (B, Nmax, 1)
        p0 = (torch.ones_like(x[:, 0, :]) / self.dim).unsqueeze(1)  # (B, 1, D)
        a1 = self.softplus(self.a1)
        a2 = self.softplus(self.a2)
        a3 = self.softplus(self.a3)
        trans = self.uot(x, p0, q0, a1, a2, a3, mask)  # (B, Nmax, D)
        frot = self.dim * x * trans * mask  # (B, Nmax, D)
        return torch.sum(frot, dim=1, keepdim=False)  # (B, D)

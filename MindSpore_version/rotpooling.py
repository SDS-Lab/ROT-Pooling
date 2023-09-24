import mindspore
import mindspore.nn as nn
#to-do
from torch_geometric.utils import softmax, to_dense_batch
from pooling.rotlayers import ROT, UOT

def to_sparse_batch(x:  mindspore.Tensor, mask:  mindspore.Tensor = None):
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
    num_nodes_graphs = mindspore.ops.zeros_like(x[:, 0, 0], dtype=mindspore.int64).fill_(n_max)
    x = x.reshape(-1, d)  # total_nodes * d
    # apply mask
    if mask is not None:
        # get number nodes per graph
        num_nodes_graphs = mask.sum(axis=1)  # bs
        # mask x
        x = x[mask.reshape(-1)]  # total_nodes * d
    # set up batch
    batch = mindspore.ops.repeat_interleave(input=mindspore.ops.arange(bs), repeats=num_nodes_graphs)
    return x, batch


class ROTPooling(nn.Cell):
    def __init__(self, dim: int, a0: float = None, a1: float = None, a2: float = None, a3: float = None,
                 rho: float = None, num: int = 4, eps: float = 1e-8, f_method: str = 'badmm-e',
                 p0: str = 'fixed', q0: str = 'fixed', same_para: bool = False):
        super(ROTPooling, self).__init__()
        self.dim = dim
        self.eps = eps
        self.num = num
        self.f_method = f_method
        self.p0 = p0
        self.q0 = q0
        self.same_para = same_para

        if rho is None:
            if self.same_para:
                self.rho = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.rho = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.rho = rho * mindspore.ops.ones(1)
            else:
                self.rho = rho * mindspore.ops.ones(self.num)

        if a0 is None:
            if self.same_para:
                self.a0 = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.a0 = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.a0 = a0 * mindspore.ops.ones(1)
            else:
                self.a0 = a0 * mindspore.ops.ones(self.num)

        if a1 is None:
            if self.same_para:
                self.a1 = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.a1 = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.a1 = a1 * mindspore.ops.ones(1)
            else:
                self.a1 = mindspore.Parameter(a1 * mindspore.ops.ones(self.num), requires_grad=True)

        if a2 is None:
            if self.same_para:
                self.a2 = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.a2 = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.a2 = a2 * mindspore.ops.ones(1)
            else:
                self.a2 = a2 * mindspore.ops.ones(self.num)

        if a3 is None:
            if self.same_para:
                self.a3 = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.a3 = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.a3 = a3 * mindspore.ops.ones(1)
            else:
                self.a3 = a3 * mindspore.ops.ones(self.num)

        self.linear_r1 = nn.Dense(in_features=self.dim, out_features=2 * self.dim, bias=False)
        self.linear_r2 = nn.Dense(in_features=2 * self.dim, out_features=1, bias=False)
        self.tanh = nn.Tanh()

        self.softplus = mindspore.ops.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(axis=1)
        self.rot = ROT(num=self.num, eps=self.eps, f_method=self.f_method)

    def construct(self, x, batch):
        """
        The feed-forward function of ROT-Pooling
        :param x: (sum_b N_b, D) samples
        :param batch: (sum_b N_b, ) the index of sets in the batch
        :return:
            a pooling result with size (K, D)
        """
        if self.q0 != 'fixed':
            q0 = softmax(self.linear_r2(self.tanh(self.linear_r1(x))), batch)  # (N, 1)
        else:
            q0 = softmax(mindspore.ops.zeros_like(x[:, 0].unsqueeze(1)), batch)  # (N, 1)
        x, mask = to_dense_batch(x, batch)  # (B, Nmax, D)
        mask = mask.unsqueeze(2)  # (B, Nmax, 1)
        q0, _ = to_dense_batch(q0, batch)  # (B, Nmax, 1)
        p0 = (mindspore.ops.ones_like(x[:, 0, :]) / self.dim).unsqueeze(1)  # (B, 1, D)
        c1 = mindspore.ops.matmul(x.permute(0, 2, 1), x) / x.shape[1]  # (B, D, D)
        c2 = mindspore.ops.matmul(x, x.permute(0, 2, 1)) / x.shape[2]  # (B, Nmax, Nmax)
        # 20221004
        c1 = c1 / mindspore.ops.max(c1)
        c2 = c2 / mindspore.ops.max(c2)
        rho = self.softplus(self.rho)
        if rho.shape[0] == 1:
            rho = rho.repeat(self.num)
        a0 = self.softplus(self.a0)
        if a0.shape[0] == 1:
            a0 = a0.repeat(self.num)

        a1 = self.softplus(self.a1)
        if a1.shape[0] == 1:
            a1 = a1.repeat(self.num)
        a2 = self.softplus(self.a2)
        if a2.shape[0] == 1:
            a2 = a2.repeat(self.num)
        a3 = self.softplus(self.a3)
        if a3.shape[0] == 1:
            a3 = a3.repeat(self.num)
        trans = self.rot(x, c1, c2, p0, q0, a0, a1, a2, a3, rho, mask)  # (B, Nmax, D)
        frot = self.dim * x * trans * mask  # (B, Nmax, D)
        return mindspore.ops.sum(frot, dim=1, keepdim=False)  # (B, D)


class UOTPooling(nn.Cell):
    def __init__(self, dim: int, a1: float = None, a2: float = None, a3: float = None, rho: float = None,
                 num: int = 4, eps: float = 1e-8, f_method: str = 'badmm-e',
                 p0: str = 'fixed', q0: str = 'fixed', same_para: bool = False):
        super(UOTPooling, self).__init__()
        self.dim = dim
        self.eps = eps
        self.num = num
        self.f_method = f_method
        self.p0 = p0
        self.q0 = q0
        self.same_para = same_para

        if rho is None:
            if self.same_para:
                self.rho = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.rho = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.rho = rho * mindspore.ops.ones(1)
            else:
                self.rho = rho * mindspore.ops.ones(self.num)

        if a1 is None:
            if self.same_para:
                self.a1 = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.a1 = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.a1 = a1 * mindspore.ops.ones(1)
            else:
                self.a1 = a1 * mindspore.ops.ones(self.num)

        if a2 is None:
            if self.same_para:
                self.a2 = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.a2 = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.a2 = a2 * mindspore.ops.ones(1)
            else:
                self.a2 = a2 * mindspore.ops.ones(self.num)

        if a3 is None:
            if self.same_para:
                self.a3 = mindspore.Parameter(0.1 * mindspore.ops.randn(1), requires_grad=True)
            else:
                self.a3 = mindspore.Parameter(0.1 * mindspore.ops.randn(self.num), requires_grad=True)
        else:
            if self.same_para:
                self.a3 = a3 * mindspore.ops.ones(1)
            else:
                self.a3 = a3 * mindspore.ops.ones(self.num)

        self.linear_r1 = nn.Dense(in_channels=self.dim, out_channels=2 * self.dim, has_bias=False)
        self.linear_r2 = nn.Dense(in_channels=2 * self.dim, out_channels=1, has_bias=False)
        self.tanh = nn.Tanh()

        self.softplus = mindspore.ops.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(axis=1)
        self.uot = UOT(num=self.num, eps=self.eps, f_method=self.f_method)

    def construct(self, x, batch):
        """
        The feed-forward function of ROT-Pooling
        :param x: (sum_b N_b, D) samples
        :param batch: (sum_b N_b, ) the index of sets in the batch
        :return:
            a pooling result with size (K, D)
        """
        if self.q0 != 'fixed':
            q0 = softmax(self.linear_r2(self.tanh(self.linear_r1(x))), batch)   # (N, 1)
        else:
            q0 = softmax(mindspore.ops.zeros_like(x[:, 0].unsqueeze(1)), batch)  # (N, 1)
        x, mask = to_dense_batch(x, batch)  # (B, Nmax, D)
        mask = mask.unsqueeze(2)  # (B, Nmax, 1)
        q0, _ = to_dense_batch(q0, batch)  # (B, Nmax, 1)
        p0 = (mindspore.ops.ones_like(x[:, 0, :]) / self.dim).unsqueeze(1)  # (B, 1, D)
        rho = self.softplus(self.rho)
        if rho.shape[0] == 1:
            rho = rho.repeat(self.num)
        a1 = self.softplus(self.a1)
        if a1.shape[0] == 1:
            a1 = a1.repeat(self.num)
        a2 = self.softplus(self.a2)
        if a2.shape[0] == 1:
            a2 = a2.repeat(self.num)
        a3 = self.softplus(self.a3)
        if a3.shape[0] == 1:
            a3 = a3.repeat(self.num)
        trans = self.uot(x, p0, q0, a1, a2, a3, rho, mask)  # (B, Nmax, D)
        frot = self.dim * x * trans * mask  # (B, Nmax, D)
        return mindspore.ops.sum(frot, dim=1, keepdim=False)  # (B, D)

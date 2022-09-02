import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

"""
Baselines (and their relations to UOT):

Set2Set (Unknown)
AveragePooling (UOT, with some alpha's)
MaxPooling (UOT, with some alpha's)
MixedPooling (UOT-barycenter, with nonparametric weights)
GatedPooling (UOT-barycenter, with parametric weights)
AttentionPooling (UOT, with one side constraint + parametric transport)
GatedAttentionPooling (UOT, with one side constraint + parametric transport)
DynamicPooling (UOT, with one side constraint + parametric transport)
DeepSet (UOT, with mixed add- and max-pooling passing through learnable parameters)
"""


class DeepSet(nn.Module):
    """
    The mixed permutation-invariant structure
    in Zaheer, Manzil, et al. "Deep Sets."
    Proceedings of the 31st International Conference on Neural Information Processing Systems. 2017.
    """
    def __init__(self, d: int, h: int):
        """
        :param d: the dimension of input samples
        :param h: the dimension of hidden representations
        """
        super(DeepSet, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1))
        self.sigmoid = nn.Sigmoid()
        self.regressor = nn.Sequential(
            nn.Linear(d, h),
            nn.ELU(inplace=True),
            nn.Linear(h, h),
            nn.ELU(inplace=True),
            nn.Linear(h, h),
            nn.ELU(inplace=True),
            nn.Linear(h, d),
        )

    def forward(self, x, batch):
        """
        :param x: (N, D) matrix
        :param batch: (N,) each element in {0, B-1}
        :return:
        """
        alpha = self.sigmoid(self.alpha)
        return self.regressor(alpha * global_add_pool(x, batch) + (1 - alpha) * global_max_pool(x, batch))


def global_softmax_pooling(x: torch.Tensor, alpha: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax to a batch of bags
    :param x: (n, d) matrix
    :param alpha: (n, 1) matrix
    :param batch: (n,) vector, each element in {0, B-1}
    :return:
        z: (B, d) matrix
    """
    alpha = softmax(alpha, batch)  # (n, 1)
    return global_add_pool(x * alpha, batch)


class MixedPooling(nn.Module):
    """
    The mixed pooling structure
    in Lee, Chen-Yu, Patrick W. Gallagher, and Zhuowen Tu.
    "Generalizing pooling functions in convolutional neural networks: Mixed, gated, and tree."
    Artificial intelligence and statistics. PMLR, 2016.
    """
    def __init__(self):
        super(MixedPooling, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, batch):
        """
        :param x: (N, D) matrix
        :param batch: (N,) each element in {0, B-1}
        :return:
        """
        alpha = self.sigmoid(self.alpha)
        return alpha * global_mean_pool(x, batch) + (1 - alpha) * global_max_pool(x, batch)


class GatedPooling(nn.Module):
    """
    The gated pooling structure
    in Lee, Chen-Yu, Patrick W. Gallagher, and Zhuowen Tu.
    "Generalizing pooling functions in convolutional neural networks: Mixed, gated, and tree."
    Artificial intelligence and statistics. PMLR, 2016.
    """
    def __init__(self, dim: int):
        super(GatedPooling, self).__init__()
        self.linear = nn.Linear(in_features=dim, out_features=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, batch):
        """
        :param x: (N, D) matrix
        :param batch: (N,) each element in {0, B-1}
        :return:
        """
        alpha = self.linear(x)  # (N, 1)
        alpha = self.sigmoid(global_add_pool(alpha, batch))
        return alpha * global_mean_pool(x, batch) + (1 - alpha) * global_max_pool(x, batch)


class Set2Set(nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class AttentionPooling(nn.Module):
    def __init__(self, d: int, h: int):
        super(AttentionPooling, self).__init__()
        self.d = d
        self.h = h
        self.linear_r1 = nn.Linear(in_features=self.d, out_features=self.h, bias=False)
        self.linear_r2 = nn.Linear(in_features=self.h, out_features=1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch):
        """
        Implement attention pooling in
        Ilse, Maximilian, Jakub Tomczak, and Max Welling.
        "Attention-based deep multiple instance learning."
        International conference on machine learning. PMLR, 2018.

        :param x: (n, d) features
        :param batch: (n,) in {0, B-1}
        :return:
            z: (B, d) features
        """
        alpha = self.linear_r2(self.tanh(self.linear_r1(x)))  # (n, 1)
        return global_softmax_pooling(x, alpha, batch)


class GatedAttentionPooling(nn.Module):
    def __init__(self, d: int, h: int):
        super(GatedAttentionPooling, self).__init__()
        self.d = d
        self.h = h
        self.linear_r1 = nn.Linear(in_features=self.d, out_features=self.h, bias=False)
        self.linear_r2 = nn.Linear(in_features=self.d, out_features=self.h, bias=False)
        self.linear_r3 = nn.Linear(in_features=self.h, out_features=1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch):
        """
        Implement gated attention pooling in
        Ilse, Maximilian, Jakub Tomczak, and Max Welling.
        "Attention-based deep multiple instance learning."
        International conference on machine learning. PMLR, 2018.

        :param x: (n, d) features
        :param batch: (n,) in {0, B-1}
        :return:
            z: (B, d) features
        """
        ux = self.tanh(self.linear_r1(x))  # (n, h)
        vx = self.softmax1(self.linear_r2(x))  # (n, h)
        alpha = self.linear_r3(ux * vx)
        return global_softmax_pooling(x, alpha, batch)


class DynamicPooling(nn.Module):
    def __init__(self, d: int, k: int = 3):
        super(DynamicPooling, self).__init__()
        self.d = d
        self.k = k
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch):
        """
        Implement the dynamic pooling layer in
        Yan, Yongluan, Xinggang Wang, Xiaojie Guo, Jiemin Fang, Wenyu Liu, and Junzhou Huang.
        "Deep multi-instance learning with dynamic pooling."
        In Asian Conference on Machine Learning, pp. 662-677. PMLR, 2018.

        :param x: (n, d) features
        :param batch: (n,) in {0, B-1}
        :return:
            z: (B, d) features
        """
        batch_size = int(batch.max().item() + 1)
        alpha = torch.zeros_like(x[:, 0]).unsqueeze(1)  # (n, 1)
        for _ in range(self.k):
            z = global_softmax_pooling(x, alpha, batch)  # (B, d)
            energy = torch.sum(z ** 2, dim=1, keepdim=True)  # (B, 1)
            z_squashed = torch.sqrt(energy) / (1 + energy) * z  # (B, d)
            for i in range(batch_size):
                alpha[batch == i, :] = alpha[batch == i, :] + torch.mm(x[batch == i, :], z_squashed[i, :].unsqueeze(1))
        return global_softmax_pooling(x, alpha, batch)


class UOTPooling(nn.Module):
    def __init__(self, d: int, h: int, k: int = 40, eps: float = 1e-8,
                 a1: float = None, a2: float = None, a3: float = None, u2: str = 'fixed'):
        super(UOTPooling, self).__init__()
        self.k = k
        self.d = d
        self.h = h
        self.eps = eps
        if a1 is None:
            self.a1 = nn.Parameter(0.1 * torch.randn(self.k), requires_grad=True)
        else:
            self.a1 = a1 * torch.ones(self.k)

        if a2 is None:
            self.a2 = nn.Parameter(0.1 * torch.randn(self.k), requires_grad=True)
        else:
            self.a2 = a2 * torch.ones(self.k)

        if a3 is None:
            self.a3 = nn.Parameter(0.1 * torch.randn(self.k), requires_grad=True)
        else:
            self.a3 = a3 * torch.ones(self.k)

        self.u2 = u2
        if self.u2 != 'fixed':
            self.linear_r = nn.Linear(in_features=self.d, out_features=d, bias=False)
        # self.linear_r1 = nn.Linear(in_features=self.d, out_features=self.h, bias=False)
        # self.linear_r2 = nn.Linear(in_features=self.h, out_features=1, bias=False)
        # self.linear_r = nn.Linear(in_features=self.d, out_features=d, bias=False)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch):
        """
        Implement a pooling operation as solving an unbalanced OT problem
        :param x: (n, d) features
        :param batch: (n,) in {0, B-1}
        :return:
            z: (B, d) features
            trans: (n, d) optimal transport matrices
        """
        batch_size = int(batch.max().item() + 1)
        a1 = self.softplus(self.a1)
        a2 = self.softplus(self.a2)
        a3 = self.softplus(self.a3)
        # a1 = self.sigmoid(self.a1)
        # a2 = self.sigmoid(self.a2)
        # a3 = self.sigmoid(self.a3)
        tmp1 = torch.zeros_like(x[:, 0].unsqueeze(1))
        # tmp1 = self.linear_r2(self.tanh(self.linear_r1(x)))  # (n, 1)
        log_u1 = torch.log(softmax(tmp1, batch))  # (n, 1)
        if self.u2 != 'fixed':
            tmp2 = global_mean_pool(self.linear_r(x), batch)  # (B, d)
            log_u2 = torch.log(self.softmax1(tmp2))  # (B, d)
        else:
            tmp2 = torch.ones_like(x[:batch_size, :]) / x.shape[1]  # (B, d)
            log_u2 = torch.log(tmp2)

        y = x / a1[0]
        for i in range(batch_size):
            b1 = torch.zeros_like(log_u1[batch == i, :])  # (n_i, 1)
            b2 = torch.zeros_like(log_u2[i, :].unsqueeze(0))  # (1, d)
            for k in range(self.k):
                log_mu1 = torch.logsumexp(y[batch == i, :], dim=1, keepdim=True)  # (n_i, 1)
                log_mu2 = torch.logsumexp(y[batch == i, :], dim=0, keepdim=True)  # (1, d)
                b1 = a2[k] / (a1[k] * (a1[k] + a2[k])) * b1 + a2[k] / (a1[k] + a2[k]) * (log_u1[batch == i, :] - log_mu1)
                b2 = a3[k] / (a1[k] * (a1[k] + a3[k])) * b2 + a3[k] / (a1[k] + a3[k]) * (log_u2[i, :].unsqueeze(0) - log_mu2)
                # print(b1.shape, b2.shape)
                y[batch == i, :] = x[batch == i, :] / a1[k] + b1 + b2
        y = torch.exp(y) + self.eps
        return x.shape[1] * global_add_pool(x * y, batch), y


class UOTPooling2(nn.Module):
    def __init__(self, d: int, h: int, k: int = 40, eps: float = 1e-8, rho: float = None,
                 a1: float = None, a2: float = None, a3: float = None, u2: str = 'fixed'):
        super(UOTPooling2, self).__init__()
        self.k = k
        self.d = d
        self.h = h
        self.eps = eps
        if rho is None:
            self.rho = nn.Parameter(0.1 * torch.randn(self.k), requires_grad=True)
        else:
            self.rho = rho * torch.ones(self.k)

        if a1 is None:
            self.a1 = nn.Parameter(0.1 * torch.randn(self.k), requires_grad=True)
        else:
            self.a1 = a1 * torch.ones(self.k)

        if a2 is None:
            self.a2 = nn.Parameter(0.1 * torch.randn(self.k), requires_grad=True)
        else:
            self.a2 = a2 * torch.ones(self.k)

        if a3 is None:
            self.a3 = nn.Parameter(0.1 * torch.randn(self.k), requires_grad=True)
        else:
            self.a3 = a3 * torch.ones(self.k)

        self.u2 = u2
        if self.u2 != 'fixed':
            self.linear_r = nn.Linear(in_features=self.d, out_features=d, bias=False)
        # self.linear_r1 = nn.Linear(in_features=self.d, out_features=self.h, bias=False)
        # self.linear_r2 = nn.Linear(in_features=self.h, out_features=1, bias=False)
        # self.linear_r = nn.Linear(in_features=self.d, out_features=d, bias=False)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, batch):
        """
        Implement a pooling operation as solving an unbalanced OT problem
        :param x: (n, d) features
        :param batch: (n,) in {0, B-1}
        :return:
            z: (B, d) features
            trans: (n, d) optimal transport matrices
        """
        batch_size = int(batch.max().item() + 1)
        a1 = self.softplus(self.a1)
        a2 = self.softplus(self.a2)
        a3 = self.softplus(self.a3)
        rho = self.softplus(self.rho)
        # a1 = self.sigmoid(self.a1)
        # a2 = self.sigmoid(self.a2)
        # a3 = self.sigmoid(self.a3)
        tmp1 = torch.zeros_like(x[:, 0].unsqueeze(1))
        # tmp1 = self.linear_r2(self.tanh(self.linear_r1(x)))  # (n, 1)
        log_u1 = torch.log(softmax(tmp1, batch))  # (n, 1)
        if self.u2 != 'fixed':
            tmp2 = global_mean_pool(self.linear_r(x), batch)  # (B, d)
            log_u2 = torch.log(self.softmax1(tmp2))  # (B, d)
        else:
            tmp2 = torch.ones_like(x[:batch_size, :]) / x.shape[1]  # (B, d)
            log_u2 = torch.log(tmp2)

        y = torch.zeros_like(x)
        for i in range(batch_size):
            log_mu1 = log_u1[batch == i, :]
            log_mu2 = log_u2[i, :].unsqueeze(0)
            z = torch.zeros_like(x[batch == i, :])
            z1 = torch.zeros_like(log_mu1)  # (n_i, 1)
            z2 = torch.zeros_like(log_mu2)  # (1, d)
            log_tran = log_mu1 + log_mu2
            log_aux = log_mu1 + log_mu2
            for k in range(self.k):
                tmp1 = log_aux + (x[batch == i, :] - z) / rho[k]
                log_tran = log_mu1 + tmp1 - torch.logsumexp(tmp1, dim=1, keepdim=True)
                tmp2 = (z + rho[k] * log_tran) / (a1[k] + rho[k])
                log_aux = tmp2 - torch.logsumexp(tmp2, dim=0, keepdim=True) + log_mu2
                # log_mu1 = log_mu1 - z1 / (rho[k] + a2[k])
                # log_mu2 = log_mu2 - z2 / (rho[k] + a3[k])
                log_mu1 = (rho[k] * log_mu1 + a2[k] * log_u1[batch == i, :] - z1) / (rho[k] + a2[k])
                log_mu2 = (rho[k] * log_mu2 + a3[k] * log_u2[i, :].unsqueeze(0) - z2) / (rho[k] + a3[k])
                tran = torch.exp(log_tran)
                aux = torch.exp(log_aux)
                z = z + a1[k] * (tran - aux)
                z1 = z1 + rho[k] * (torch.exp(log_mu1) - torch.sum(tran, dim=1, keepdim=True))
                z2 = z2 + rho[k] * (torch.exp(log_mu2) - torch.sum(aux, dim=0, keepdim=True))
            y[batch == i, :] = torch.exp(log_tran) + self.eps
        return x.shape[1] * global_add_pool(x * y, batch), y


def uot_sinkhorn(x: torch.Tensor, log_u1: torch.Tensor, log_u2: torch.Tensor,
                 a1: float, a2: float, a3: float, k: int, cutting: bool = False) -> torch.Tensor:
    """
    Sinkhorn Scaling algorithm
    :param x: a (N, D) matrix
    :param log_u1: (N, 1) log of parametrized side constraint
    :param log_u2: (1, D) log of parametrized side constraint
    :param a1: (1, 1) parametrized hyperparameter
    :param a2: (1, 1) parametrized hyperparameter
    :param a3: (1, 1) parametrized hyperparameter
    :param k: the number of sinkhorn iterations
    :param cutting: set a1 == 1 or not
    :return:
    """
    y = x / a1

    b1 = torch.zeros_like(log_u1)
    b2 = torch.zeros_like(log_u2)
    # print(b1.shape, b2.shape)
    for _ in range(k):
        log_mu1 = torch.logsumexp(y, dim=1, keepdim=True)
        log_mu2 = torch.logsumexp(y, dim=0, keepdim=True)
        b1 = a2 / (a1 + a2) * (b1 / a1 + log_u1 - log_mu1)
        b2 = a3 / (a1 + a3) * (b2 / a1 + log_u2 - log_mu2)
        # print(b1.shape, b2.shape)
        y = x / a1 + b1 + b2
    return torch.exp(y)


def pga_sinkhorn(x: torch.Tensor, log_u1: torch.Tensor, log_u2: torch.Tensor, a0: float,
                 a1: float, a2: float, a3: float, t: int, k: int, rho: float = 1) -> torch.Tensor:
    """
    Sinkhorn Scaling algorithm
    :param x: a (N, D) matrix
    :param log_u1: (N, 1) log of parametrized side constraint
    :param log_u2: (1, D) log of parametrized side constraint
    :param a0: (1, 1) parametrized hyperparameter
    :param a1: (1, 1) parametrized hyperparameter
    :param a2: (1, 1) parametrized hyperparameter
    :param a3: (1, 1) parametrized hyperparameter
    :param t: the number of proximal step
    :param k: the number of sinkhorn iterations
    :param rho: the parameters of Lagrangian term
    :param eps: a small offset for numerical stability
    :return:
    """
    mu1 = torch.mean(x, dim=0, keepdim=True)  # (1, D)
    mu2 = torch.mean(x, dim=1, keepdim=True)  # (N, 1)
    c2 = torch.t(x - mu1) @ (x - mu1) / x.shape[0]  # (D, D)
    c1 = (x - mu2) @ torch.t(x - mu2) / x.shape[1]  # (N, N)

    log_trans = log_u1 + log_u2
    trans = torch.exp(log_trans)
    for _ in range(t):
        cost = -x - a0 * c1 @ trans @ c2 - rho * log_trans
        y = -cost / (a1 + rho)
        b1 = torch.zeros_like(log_u1)
        b2 = torch.zeros_like(log_u2)
        # print(b1.shape, b2.shape)
        for _ in range(k):
            log_mu1 = torch.logsumexp(y, dim=1, keepdim=True)
            log_mu2 = torch.logsumexp(y, dim=0, keepdim=True)
            b1 = a2 / (a1 + rho + a2) * (b1 / (a1 + rho) + log_u1 - log_mu1)
            b2 = a3 / (a1 + rho + a3) * (b2 / (a1 + rho) + log_u2 - log_mu2)
            # print(b1.shape, b2.shape)
            y = -cost / (a1 + rho) + b1 + b2
        log_trans = y
        trans = torch.exp(log_trans)
    return trans


def uot_badmm(x: torch.Tensor, log_u1: torch.Tensor, log_u2: torch.Tensor, a1: float, a2: float,
              a3: float, k: int, rho: float = 1) -> torch.Tensor:
    """
    Bregman ADMM algorithm for unbalanced optimal transport
    :param x: a (N, D) matrix
    :param log_u1: (N, 1) log of parametrized side constraint
    :param log_u2: (1, D) log of parametrized side constraint
    :param a1: (1, 1) parametrized hyperparameter
    :param a2: (1, 1) parametrized hyperparameter
    :param a3: (1, 1) parametrized hyperparameter
    :param k: the number of sinkhorn iterations
    :param rho: the parameters of Lagrangian term
    :param eps: a small offset for numerical stability
    :return:
    """
    z = torch.zeros_like(x)
    z1 = torch.zeros_like(log_u1)
    z2 = torch.zeros_like(log_u2)
    log_u10 = log_u1
    log_u20 = log_u2
    log_tran = log_u1 + log_u2
    log_aux = log_u1 + log_u2
    for _ in range(k):
        tmp1 = log_aux + (x - z) / rho
        log_tran = log_u1 + tmp1 - torch.logsumexp(tmp1, dim=1, keepdim=True)
        tmp2 = (z + rho * log_tran) / (a1 + rho)
        log_aux = tmp2 - torch.logsumexp(tmp2, dim=0, keepdim=True) + log_u2
        tmp3 = (rho * log_u1 + a2 * log_u10 - z1) / (rho + a2)  # (N, 1)
        log_u1 = tmp3 - torch.logsumexp(tmp3, dim=0, keepdim=True)
        tmp4 = (rho * log_u2 + a3 * log_u20 - z2) / (rho + a3)  # (1, D)
        log_u2 = tmp4 - torch.logsumexp(tmp4, dim=1, keepdim=True)
        tran = torch.exp(log_tran)
        aux = torch.exp(log_aux)
        z = z + rho * (tran - aux)
        z1 = z1 + rho * (torch.exp(log_u1) - torch.sum(tran, dim=1, keepdim=True))
        z2 = z2 + rho * (torch.exp(log_u2) - torch.sum(aux, dim=0, keepdim=True))
    return torch.exp(log_tran)


def pga_badmm(x: torch.Tensor, log_u1: torch.Tensor, log_u2: torch.Tensor,
              a0: float, a1: float, a2: float, a3: float, k: int, rho: float = 1) -> torch.Tensor:
    """
    Bregman ADMM algorithm for unbalanced optimal transport
    :param x: a (N, D) matrix
    :param log_u1: (N, 1) log of parametrized side constraint
    :param log_u2: (1, D) log of parametrized side constraint
    :param a0: (1, 1) parametrized hyperparameter
    :param a1: (1, 1) parametrized hyperparameter
    :param a2: (1, 1) parametrized hyperparameter
    :param a3: (1, 1) parametrized hyperparameter
    :param k: the number of sinkhorn iterations
    :param rho: the parameters of Lagrangian term
    :param eps: a small offset for numerical stability
    :return:
    """
    z = torch.zeros_like(x)
    z1 = torch.zeros_like(log_u1)
    z2 = torch.zeros_like(log_u2)
    log_u10 = log_u1
    log_u20 = log_u2
    log_tran = log_u1 + log_u2
    log_aux = log_u1 + log_u2

    mu1 = torch.mean(x, dim=0, keepdim=True)  # (1, D)
    mu2 = torch.mean(x, dim=1, keepdim=True)  # (N, 1)
    c2 = torch.t(x - mu1) @ (x - mu1) / x.shape[0]  # (D, D)
    c1 = (x - mu2) @ torch.t(x - mu2) / x.shape[1]  # (N, N)

    tran = torch.exp(log_tran)
    aux = torch.exp(log_aux)

    for _ in range(k):
        tmp1 = log_aux + (x - a0 * c1 @ aux @ c2 - z) / rho
        log_tran = log_u1 + tmp1 - torch.logsumexp(tmp1, dim=1, keepdim=True)

        tmp2 = (z + rho * log_tran + a0 * c1 @ tran @ c2) / (a1 + rho)
        log_aux = tmp2 - torch.logsumexp(tmp2, dim=0, keepdim=True) + log_u2

        tmp3 = (rho * log_u1 + a2 * log_u10 - z1) / (rho + a2)  # (N, 1)
        log_u1 = tmp3 - torch.logsumexp(tmp3, dim=0, keepdim=True)

        tmp4 = (rho * log_u2 + a3 * log_u20 - z2) / (rho + a3)  # (1, D)
        log_u2 = tmp4 - torch.logsumexp(tmp4, dim=1, keepdim=True)

        tran = torch.exp(log_tran)
        aux = torch.exp(log_aux)

        z = z + rho * (tran - aux)
        z1 = z1 + rho * (torch.exp(log_u1) - torch.sum(tran, dim=1, keepdim=True))
        z2 = z2 + rho * (torch.exp(log_u2) - torch.sum(aux, dim=0, keepdim=True))
    return tran


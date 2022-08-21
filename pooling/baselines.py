import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

"""
AveragePooling (UOT, with some alpha's)
MaxPooling (UOT, with some alpha's)
GeneralizedNormPooling
DeepSet (UOT, with mixed add- and max-pooling passing through learnable parameters)
MixedPooling (UOT-barycenter, with nonparametric weights)
GatedPooling (UOT-barycenter, with parametric weights)
Set2Set
AttentionPooling (UOT, with one side constraint + parametric transport)
GatedAttentionPooling (UOT, with one side constraint + parametric transport)
DynamicPooling (UOT, with one side constraint + parametric transport)
"""


class GeneralizedNormPooling(nn.Module):
    def __init__(self, d: int):
        super(GeneralizedNormPooling, self).__init__()
        self.ps = nn.Parameter(torch.randn(2))
        self.qs = nn.Parameter(torch.randn(2))
        self.softplus = nn.Softplus()
        self.thres = nn.Threshold(threshold=-50, value=50, inplace=False)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(in_features=d, out_features=d, bias=True)
        self.epsilon = 1e-6

    def forward(self, x, batch):
        """
        :param x: (N, D) matrix
        :param batch: (N,) each element in {0, B-1}
        :return:
            z: (B, d) matrix
        """
        nums = torch.ones_like(x[:, 0])
        nums = global_add_pool(nums, batch=batch).unsqueeze(1)  # (B, 1)

        ps = -self.thres(-self.softplus(self.ps))
        qs = self.tanh(self.qs)

        x = torch.abs(x) + self.epsilon
        d = x.shape[0]
        d1 = int(d/2)

        x_pos = torch.exp(ps[0] * torch.log(x[:, :d1]))
        gnp_pos = torch.exp(torch.log(global_add_pool(x_pos, batch=batch)) / ps[0]) / (nums ** qs[0])

        x_neg = x[:, d1:]
        gnp_neg = (global_add_pool(x_neg ** ps[1], batch=batch) ** (1 / ps[1])) / (nums ** qs[0])
        gnp = self.linear(torch.cat((gnp_pos, gnp_neg), dim=1))
        return gnp


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

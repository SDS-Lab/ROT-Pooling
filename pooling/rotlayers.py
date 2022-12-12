import torch
import torch.nn as nn


def uot_sinkhorn(x: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
                 a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, mask: torch.Tensor,
                 num: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """
    Solving regularized optimal transport via Sinkhorn-scaling
    :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
    :param p0: (B, 1, D), the marginal prior of dimensions
    :param q0: (B, N, 1), the marginal prior of samples
    :param a1: (num, ), the weight of the entropic term
    :param a2: (num, ), the weight of the KL term of p0
    :param a3: (num, ), the weight of the KL term of q0
    :param mask: (B, N, 1) a masking tensor
    :param num: the number of outer iterations
    :param eps: the epsilon to avoid numerical instability
    :return:
        t: (B, N, D), the optimal transport matrix
    """
    t = (q0 * p0) * mask  # （B, N, D）
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps) * mask  # (B, N, 1)
    tau = 0.0
    cost = (-x - tau * torch.log(t + eps)) * mask  # (B, N, D)
    a = torch.zeros_like(p0)  # (B, 1, D)
    b = torch.zeros_like(q0)  # (B, N, 1)
    a11 = a1[0] + tau
    y = -cost / a11  # (B, N, D)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        a11 = a1[n] + tau
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        log_p = torch.log(torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) + ymax   # (B, 1, D)
        log_q = torch.logsumexp(y, dim=2, keepdim=True) * mask  # (B, N, 1)
        a = a2[n] / (a2[n] + a11) * (a / a11 + log_p0 - log_p)
        b = a3[n] / (a3[n] + a11) * (b / a11 + log_q0 - log_q)
        y = (-cost / a11 + a + b) * mask
    t = torch.exp(y) * mask
    return t


def rot_sinkhorn(x: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
                 a0: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, mask: torch.Tensor,
                 num: int = 4, inner: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """
    Solving regularized optimal transport via Sinkhorn-scaling
    :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
    :param c1: (B, D, D), a matrix with size D x D
    :param c2: (B, N, N), a matrix with size N x N
    :param p0: (B, 1, D), the marginal prior of dimensions
    :param q0: (B, N, 1), the marginal prior of samples
    :param a0: (num, ), the weight of the GW term
    :param a1: (num, ), the weight of the entropic term
    :param a2: (num, ), the weight of the KL term of p0
    :param a3: (num, ), the weight of the KL term of q0
    :param mask: (B, N, 1) a masking tensor
    :param num: the number of outer iterations
    :param inner: the number of inner Sinkhorn iterations
    :param eps: the epsilon to avoid numerical instability
    :return:
        t: (B, N, D), the optimal transport matrix
    """
    t = (q0 * p0) * mask  # （B, N, D）
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps) * mask  # (B, N, 1)
    tau = 1.0
    for m in range(num):
        n = min([m, a1.shape[0]-1])
        a11 = a1[n] + tau
        tmp1 = torch.matmul(c2, t)  # (B, N, D)
        tmp2 = torch.matmul(tmp1, c1)  # (B, N, D)
        cost = (-x - a0[n] * tmp2 - tau * torch.log(t + eps)) * mask  # (B, N, D)
        a = torch.zeros_like(p0)  # (B, 1, D)
        b = torch.zeros_like(q0)  # (B, N, 1)
        y = -cost / a11  # (B, N, D)
        for k in range(inner):
            ymin, _ = torch.min(y, dim=1, keepdim=True)
            ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
            log_p = torch.log(torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) + ymax  # (B, 1, D)
            log_q = torch.logsumexp(y, dim=2, keepdim=True) * mask   # (B, N, 1)
            a = a2[n] / (a2[n] + a11) * (a / a11 + log_p0 - log_p)
            b = a3[n] / (a3[n] + a11) * (b / a11 + log_q0 - log_q)
            y = (-cost / a11 + a + b) * mask
        t = torch.exp(y) * mask
    return t


def uot_badmm(x: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
              a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, rho: torch.Tensor,
              mask: torch.Tensor, num: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """
    Solving regularized optimal transport via Bregman ADMM algorithm (entropic regularizer)
    :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
    :param p0: (B, 1, D), the marginal prior of dimensions
    :param q0: (B, N, 1), the marginal prior of samples
    :param a1: (num, ), the weight of the entropic term
    :param a2: (num, ), the weight of the KL term of p0
    :param a3: (num, ), the weight of the KL term of q0
    :param rho: (num, ), the learning rate of ADMM
    :param mask: (B, N, 1) a masking tensor
    :param num: the number of Bregman ADMM iterations
    :param eps: the epsilon to avoid numerical instability
    :return:
        t: (N, D), the optimal transport matrix
    """
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps) * mask  # (B, N, 1)
    log_t = (log_q0 + log_p0) * mask  # (B, N, D)
    log_s = (log_q0 + log_p0) * mask  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps) * mask  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        # update logP
        y = ((x - z) / rho[n] + log_s)  # (B, N, D)
        log_t = mask * (log_eta - torch.logsumexp(y, dim=2, keepdim=True)) + y  # (B, N, D)
        # update logS
        y = (z + rho[n] * log_t) / (a1[n] + rho[n])  # (B, N, D)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin- mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        # (B, N, D)
        log_s = mask * (
                log_mu - torch.log(torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) + y
        # update dual variables
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)  # (B, 1, D)
        y = ((rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n]))  # (B, N, 1)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        log_eta = (y - torch.log(
            torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) * mask  # (B, N, 1)
        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(s, dim=1, keepdim=True))  # (B, 1, D)
        z2 = z2 + rho[n] * (torch.exp(log_eta) * mask - torch.sum(t, dim=2, keepdim=True)) * mask  # (B, N, 1)
    return torch.exp(log_t) * mask


def rot_badmm(x: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
              a0: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, rho: torch.Tensor,
              mask: torch.Tensor, num: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """
    Solving regularized optimal transport via Bregman ADMM algorithm (entropic regularizer)
    :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
    :param c1: (B, D, D), a matrix with size D x D
    :param c2: (B, N, N), a matrix with size N x N
    :param p0: (B, 1, D), the marginal prior of dimensions
    :param q0: (B, N, 1), the marginal prior of samples
    :param a0: (num, ), the weight of the GW term
    :param a1: (num, ), the weight of the entropic term
    :param a2: (num, ), the weight of the KL term of p0
    :param a3: (num, ), the weight of the KL term of q0
    :param rho: (num, ), the learning rate of ADMM
    :param mask: (B, N, 1) a masking tensor
    :param num: the number of Bregman ADMM iterations
    :param eps: the epsilon to avoid numerical instability
    :return:
        t: (N, D), the optimal transport matrix
    """
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps) * mask  # (B, N, 1)
    log_t = (log_q0 + log_p0) * mask  # (B, N, D)
    log_s = (log_q0 + log_p0) * mask  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps) * mask  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        # update logP
        tmp1 = torch.matmul(c2, torch.exp(log_s) * mask)
        tmp2 = torch.matmul(tmp1, c1)
        y = (x + a0[n] * tmp2 - z) / rho[n] + log_s  # (B, N, D)
        log_t = mask * (log_eta - torch.logsumexp(y, dim=2, keepdim=True)) + y  # (B, N, D)
        # update logS
        tmp1 = torch.matmul(c2, torch.exp(log_t) * mask)
        tmp2 = torch.matmul(tmp1, c1)
        y = (z + a0[n] * tmp2 + rho[n] * log_t) / (a1[n] + rho[n])  # (B, N, D)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        # (B, N, D)
        log_s = mask * (
                log_mu - torch.log(torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) + y
        # update dual variables
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)
        # update log_mu
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)  # (B, 1, D)
        # update log_eta
        y = ((rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n])) * mask  # (B, N, 1)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        log_eta = (y - torch.log(
            torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) * mask  # (B, N, 1)
        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(s, dim=1, keepdim=True))  # (B, 1, D)
        z2 = z2 + rho[n] * (torch.exp(log_eta) * mask - torch.sum(t, dim=2, keepdim=True)) * mask  # (B, N, 1)
    return torch.exp(log_t) * mask


def uot_badmm2(x: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
               a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, rho: torch.Tensor,
               mask: torch.Tensor, num: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """
    Solving regularized optimal transport via Bregman ADMM algorithm (quadratic regularizer)
    :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
    :param p0: (B, 1, D), the marginal prior of dimensions
    :param q0: (B, N, 1), the marginal prior of samples
    :param a1: (num, ), the weight of the entropic term
    :param a2: (num, ), the weight of the KL term of p0
    :param a3: (num, ), the weight of the KL term of q0
    :param rho: (num, ), the learning rate of ADMM
    :param mask: (B, N, 1) a masking tensor
    :param num: the number of Bregman ADMM iterations
    :param eps: the epsilon to avoid numerical instability
    :param rho: the learning rate of ADMM
    :return:
        t: (N, D), the optimal transport matrix
    """
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps) * mask  # (B, N, 1)
    log_t = (log_q0 + log_p0) * mask  # (B, N, D)
    log_s = (log_q0 + log_p0) * mask  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps) * mask  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        # update logP
        y = (x - a1[n] * torch.exp(log_s) * mask - z) / rho[n] + log_s  # (B, N, D)
        log_t = mask * (log_eta - torch.logsumexp(y, dim=2, keepdim=True)) + y  # (B, N, D)
        # update logS
        y = (z - a1[n] * torch.exp(log_t) * mask) / rho[n] + log_t  # (B, N, D)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        # (B, N, D)
        log_s = mask * (
                log_mu - torch.log(torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) + y
        # update dual variables
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)
        # update log_mu
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)  # (B, 1, D)
        # update log_eta
        y = ((rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n])) * mask  # (B, N, 1)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        log_eta = (y - torch.log(
            torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) * mask  # (B, N, 1)
        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(s, dim=1, keepdim=True))  # (B, 1, D)
        z2 = z2 + rho[n] * (torch.exp(log_eta) * mask - torch.sum(t, dim=2, keepdim=True)) * mask  # (B, N, 1)
    return torch.exp(log_t) * mask


def rot_badmm2(x: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
               a0: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, rho: torch.Tensor,
               mask: torch.Tensor, num: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """
    Solving regularized optimal transport via Bregman ADMM algorithm (quadratic regularizer)
    :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
    :param c1: (B, D, D), a matrix with size D x D
    :param c2: (B, N, N), a matrix with size N x N
    :param p0: (B, 1, D), the marginal prior of dimensions
    :param q0: (B, N, 1), the marginal prior of samples
    :param a0: (num, ), the weight of the GW term
    :param a1: (num, ), the weight of the entropic term
    :param a2: (num, ), the weight of the KL term of p0
    :param a3: (num, ), the weight of the KL term of q0
    :param rho: (num, ), the weight of the ADMM term
    :param mask: (B, N, 1) a masking tensor
    :param num: the number of Bregman ADMM iterations
    :param eps: the epsilon to avoid numerical instability
    :param rho: the learning rate of ADMM
    :return:
        t: (N, D), the optimal transport matrix
    """
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps) * mask  # (B, N, 1)
    log_t = (log_q0 + log_p0) * mask  # (B, N, D)
    log_s = (log_q0 + log_p0) * mask  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps) * mask  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        # update logP
        tmp1 = torch.matmul(c2, torch.exp(log_s) * mask)
        tmp2 = torch.matmul(tmp1, c1)
        y = (x + a0[n] * tmp2 - a1[n] * torch.exp(log_s) * mask - z) / rho[n] + log_s  # (B, N, D)
        log_t = mask * (log_eta - torch.logsumexp(y, dim=2, keepdim=True)) + y
        # update logS
        tmp1 = torch.matmul(c2, torch.exp(log_t) * mask)
        tmp2 = torch.matmul(tmp1, c1)
        y = (z + a0[n] * tmp2 - a1[n] * torch.exp(log_t) * mask) / rho[n] + log_t  # (B, N, D)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        # (B, N, D)
        log_s = mask * (
                log_mu - torch.log(torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) + y
        # update dual variables
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)
        # update log_mu
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)  # (B, 1, D)
        # update log_eta
        y = ((rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n])) * mask  # (B, N, 1)
        ymin, _ = torch.min(y, dim=1, keepdim=True)
        ymax, _ = torch.max(ymin - mask * ymin + y, dim=1, keepdim=True)  # (B, 1, D)
        log_eta = (y - torch.log(
            torch.sum(torch.exp((y - ymax) * mask) * mask, dim=1, keepdim=True)) - ymax) * mask  # (B, N, 1)
        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(s, dim=1, keepdim=True))  # (B, 1, D)
        z2 = z2 + rho[n] * (torch.exp(log_eta) * mask - torch.sum(t, dim=2, keepdim=True)) * mask  # (B, N, 1)
    return torch.exp(log_t) * mask


class ROT(nn.Module):
    """
    Neural network layer to implement regularized optimal transport.

    Parameters:
    -----------
    :param num: int, the number of iterations
    :param eps: float, default: 1.0e-8
        The epsilon avoiding numerical instability
    :param f_method: str, default: 'badmm-e'
        The feed-forward method, badmm-e, badmm-q, or sinkhorn
    """

    def __init__(self, num: int = 4, eps: float = 1e-8, f_method: str = 'badmm-e'):
        super(ROT, self).__init__()
        self.num = num
        self.eps = eps
        self.f_method = f_method

    def forward(self, x, c1, c2, p0, q0, a0, a1, a2, a3, rho, mask):
        """
        Solving regularized OT problem
        """
        if self.f_method == 'badmm-e':
            t = rot_badmm(x, c1, c2, p0, q0, a0, a1, a2, a3, rho, mask, self.num, self.eps)
        elif self.f_method == 'badmm-q':
            t = rot_badmm2(x, c1, c2, p0, q0, a0, a1, a2, a3, rho, mask, self.num, self.eps)
        else:
            t = rot_sinkhorn(x, c1, c2, p0, q0, a0, a1, a2, a3, mask, self.num, inner=0, eps=self.eps)
        return t


class UOT(nn.Module):
    """
    Neural network layer to implement unbalanced optimal transport.

    Parameters:
    -----------
    :param num: int, the number of iterations
    :param eps: float, default: 1.0e-8
        The epsilon avoiding numerical instability
    :param f_method: str, default: 'badmm-e'
        The feed-forward method, badmm-e, badmm-q or sinkhorn
    """

    def __init__(self, num: int = 4, eps: float = 1e-8, f_method: str = 'badmm-e'):
        super(UOT, self).__init__()
        self.num = num
        self.eps = eps
        self.f_method = f_method

    def forward(self, x, p0, q0, a1, a2, a3, rho, mask):
        """
        Solving regularized OT problem
        """
        if self.f_method == 'badmm-e':
            t = uot_badmm(x, p0, q0, a1, a2, a3, rho, mask, self.num, self.eps)
        elif self.f_method == 'badmm-q':
            t = uot_badmm2(x, p0, q0, a1, a2, a3, rho, mask, self.num, self.eps)
        else:
            t = uot_sinkhorn(x, p0, q0, a1, a2, a3, mask, self.num, eps=self.eps)
        return t

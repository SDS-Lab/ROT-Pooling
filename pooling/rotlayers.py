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
    t = q0 * p0  # （B, N, D）
    log_p0 = torch.log(p0 + eps)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps)  # (B, N, 1)
    tau = 1.0
    cost = -x - tau * torch.log(t + eps)  # (B, N, D)
    a = torch.zeros_like(p0)  # (B, 1, D)
    b = torch.zeros_like(q0)  # (B, N, 1)
    a11 = a1[0] + tau
    y = -cost / a11  # (B, N, D)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        a11 = a1[n] + tau
        log_p = torch.logsumexp(y, dim=1, keepdim=True)   # (B, 1, D)
        log_q = torch.logsumexp(y, dim=2, keepdim=True)   # (B, N, 1)
        a = a2[n] / (a2[n] + a11) * (a / a11 + log_p0 - log_p)
        b = a3[n] / (a3[n] + a11) * (b / a11 + log_q0 - log_q)
        y = -cost / a11 + a + b

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
    t = q0 * p0  # （B, N, D）
    log_p0 = torch.log(p0 + eps)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps)  # (B, N, 1)
    tau = 1.0
    for m in range(num):
        n = min([m, a1.shape[0]-1])
        a11 = a1[n] + tau
        tmp1 = torch.bmm(c2, t)  # (B, N, D)
        tmp2 = torch.bmm(tmp1, c1)  # (B, N, D)
        cost = -x - a0[n] * tmp2 - tau * torch.log(t + eps)  # (B, N, D)
        a = torch.zeros_like(p0)  # (B, 1, D)
        b = torch.zeros_like(q0)  # (B, N, 1)
        y = -cost / a11  # (B, N, D)
        for k in range(inner):
            log_p = torch.logsumexp(y, dim=1, keepdim=True)   # (B, 1, D)
            log_q = torch.logsumexp(y, dim=2, keepdim=True)   # (B, N, 1)
            a = a2[n] / (a2[n] + a11) * (a / a11 + log_p0 - log_p)
            b = a3[n] / (a3[n] + a11) * (b / a11 + log_q0 - log_q)
            y = -cost / a11 + a + b
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
    log_t = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_s = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps)  # (B, N, 1)
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps)  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    d1 = torch.ones_like(p0)  # (B, 1, D)
    n1 = torch.ones_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        # update logP
        y = (x - z) / rho[n] + log_s  # (B, N, D)
        log_t = n1 * (log_mu - torch.logsumexp(y, dim=2, keepdim=True)) + y

        # update logS
        y = (z + rho[n] * log_t) / (a1[n] + rho[n])  # (B, N, D)
        log_s = (log_eta - torch.logsumexp(y, dim=1, keepdim=True)) * d1 + y

        # update dual variables
        #cheng changed
        t_temp = torch.exp(log_t)
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)

        # update log_mu
        # log_mu2 = torch.log(torch.sum(s, dim=1, keepdim=True) + eps)
        # y = (rho * log_mu + rho * log_mu2 + a2 * log_p0 - 2 * z1) / (2 * rho + a2)  # (B, 1, D)
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)

        # update log_eta
        # log_eta2 = torch.log(torch.sum(t, dim=2, keepdim=True) + eps)
        # y = (rho * log_eta + rho * log_eta2 + a3 * log_q0 - 2 * z2) / (2 * rho + a3)  # (B, N, 1)
        y = (rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n])  # (B, N, 1)
        log_eta = y - torch.logsumexp(y, dim=1, keepdim=True)

        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(t, dim=2, keepdim=True))
        z2 = z2 + rho[n] * (torch.exp(log_eta) - torch.sum(s, dim=1, keepdim=True))
    temp = torch.exp(log_t) * mask
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
    log_t = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_s = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps)  # (B, N, 1)
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps)  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    d1 = torch.ones_like(p0)  # (B, 1, D)
    n1 = torch.ones_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        # update logP
        tmp1 = torch.bmm(c2, torch.exp(log_s))
        tmp2 = torch.bmm(tmp1, c1)
        y = (x + a0[n] * tmp2 - z) / rho[n] + log_s  # (B, N, D)
        log_t = n1 * (log_mu - torch.logsumexp(y, dim=2, keepdim=True)) + y

        # update logS
        tmp1 = torch.bmm(c2, torch.exp(log_t))
        tmp2 = torch.bmm(tmp1, c1)
        y = (z + a0[n] * tmp2 + rho[n] * log_t) / (a1[n] + rho[n])  # (B, N, D)
        log_s = (log_eta - torch.logsumexp(y, dim=1, keepdim=True)) * d1 + y

        # update dual variables
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)

        # update log_mu
        # log_mu2 = torch.log(torch.sum(s, dim=1, keepdim=True) + eps)
        # y = (rho * log_mu + rho * log_mu2 + a2 * log_p0 - 2 * z1) / (2 * rho + a2)  # (B, 1, D)
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)

        # update log_eta
        # log_eta2 = torch.log(torch.sum(t, dim=2, keepdim=True) + eps)
        # y = (rho * log_eta + rho * log_eta2 + a3 * log_q0 - 2 * z2) / (2 * rho + a3)  # (B, N, 1)
        y = (rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n])  # (B, N, 1)
        log_eta = y - torch.logsumexp(y, dim=1, keepdim=True)

        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(t, dim=2, keepdim=True))
        z2 = z2 + rho[n] * (torch.exp(log_eta) - torch.sum(s, dim=1, keepdim=True))

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
    s = q0 * p0
    t = q0 * p0
    log_t = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_s = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_mu = torch.log(p0)  # (B, 1, D)
    log_eta = torch.log(q0 + eps)  # (B, N, 1)
    log_p0 = torch.log(p0)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps)  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    d1 = torch.ones_like(p0)  # (B, 1, D)
    n1 = torch.ones_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])

        # update logP
        y = (x - a1[n] * s - z) / rho[n] + log_s  # (B, N, D)
        log_t = n1 * (log_mu - torch.logsumexp(y, dim=2, keepdim=True)) + y

        # update logS
        y = (z - a1[n] * t) / rho[n] + log_t  # (B, N, D)
        log_s = (log_eta - torch.logsumexp(y, dim=1, keepdim=True)) * d1 + y

        # update dual variables
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)

        # update log_mu
        # log_mu2 = torch.log(torch.sum(s, dim=1, keepdim=True) + eps)
        # y = (rho * log_mu + rho * log_mu2 + a2 * log_p0 - 2 * z1) / (2 * rho + a2)  # (B, 1, D)
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)

        # update log_eta
        # log_eta2 = torch.log(torch.sum(t, dim=2, keepdim=True) + eps)
        # y = (rho * log_eta + rho * log_eta2 + a3 * log_q0 - 2 * z2) / (2 * rho + a3)  # (B, N, 1)
        y = (rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n])  # (B, N, 1)
        log_eta = y - torch.logsumexp(y, dim=1, keepdim=True)

        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(t, dim=2, keepdim=True))
        z2 = z2 + rho[n] * (torch.exp(log_eta) - torch.sum(s, dim=1, keepdim=True))

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
    s = q0 * p0
    t = q0 * p0
    log_t = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_s = torch.log(q0 * p0 + eps)  # (B, N, D)
    log_mu = torch.log(p0 + eps)  # (B, 1, D)
    log_eta = torch.log(q0 + eps)  # (B, N, 1)
    log_p0 = torch.log(p0 + eps)  # (B, 1, D)
    log_q0 = torch.log(q0 + eps)  # (B, N, 1)
    z = torch.zeros_like(log_t)  # (B, N, D)
    z1 = torch.zeros_like(p0)  # (B, 1, D)
    z2 = torch.zeros_like(q0)  # (B, N, 1)
    d1 = torch.ones_like(p0)  # (B, 1, D)
    n1 = torch.ones_like(q0)  # (B, N, 1)
    for k in range(num):
        n = min([k, a1.shape[0] - 1])
        # update logP
        tmp1 = torch.bmm(c2, torch.exp(log_s))
        tmp2 = torch.bmm(tmp1, c1)
        y = (x + a0[n] * tmp2 - a1[n] * s - z) / rho[n] + log_s  # (B, N, D)
        log_t = n1 * (log_mu - torch.logsumexp(y, dim=2, keepdim=True)) + y

        # update logS
        tmp1 = torch.bmm(c2, torch.exp(log_t))
        tmp2 = torch.bmm(tmp1, c1)
        y = (z + a0[n] * tmp2 - a1[n] * t) / rho[n] + log_t  # (B, N, D)
        log_s = (log_eta - torch.logsumexp(y, dim=1, keepdim=True)) * d1 + y

        # update dual variables
        t = torch.exp(log_t) * mask
        s = torch.exp(log_s) * mask
        z = z + rho[n] * (t - s)

        # update log_mu
        # log_mu2 = torch.log(torch.sum(s, dim=1, keepdim=True) + eps)
        # y = (rho * log_mu + rho * log_mu2 + a2 * log_p0 - 2 * z1) / (2 * rho + a2)  # (B, 1, D)
        y = (rho[n] * log_mu + a2[n] * log_p0 - z1) / (rho[n] + a2[n])  # (B, 1, D)
        log_mu = y - torch.logsumexp(y, dim=2, keepdim=True)

        # update log_eta
        # log_eta2 = torch.log(torch.sum(t, dim=2, keepdim=True) + eps)
        # y = (rho * log_eta + rho * log_eta2 + a3 * log_q0 - 2 * z2) / (2 * rho + a3)  # (B, N, 1)
        y = (rho[n] * log_eta + a3[n] * log_q0 - z2) / (rho[n] + a3[n])  # (B, N, 1)
        log_eta = y - torch.logsumexp(y, dim=1, keepdim=True)

        # update dual variables
        z1 = z1 + rho[n] * (torch.exp(log_mu) - torch.sum(t, dim=2, keepdim=True))
        z2 = z2 + rho[n] * (torch.exp(log_eta) - torch.sum(s, dim=1, keepdim=True))

    return torch.exp(log_t) * mask


class RegularizedOptimalTransport(torch.autograd.Function):
    """
    PyTorch autograd function for entropy regularized optimal transport. Assumes batched inputs as follows:
    Allows for approximate gradient calculations, which is faster and may be useful during early stages of learning,
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, p0: torch.Tensor, q0: torch.Tensor,
                a0: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, mask: torch.Tensor,
                num: int = 4, eps: float = 1e-8, f_method: float = 'badmm'):
        """
            Solving regularized optimal transport via Bregman ADMM algorithm or Sinkhorn-scaling algorithm
            :param ctx
            :param x: (B, N, D), a matrix with N samples and each sample is D-dimensional
            :param c1: (B, D, D), a matrix with size D x D
            :param c2: (B, N, N), a matrix with size N x N
            :param p0: (B, 1, D), the marginal prior of dimensions
            :param q0: (B, N, 1), the marginal prior of samples
            :param a0: (1, ), the weight of the GW term
            :param a1: (1, ), the weight of the entropic term
            :param a2: (1, ), the weight of the KL term of p0
            :param a3: (1, ), the weight of the KL term of q0
            :param mask: (B, N, 1) a masking tensor
            :param num: the number of Bregman ADMM iterations
            :param eps: the epsilon to avoid numerical instability
            :param f_method: the feed-forward methods, Bregman ADMM or Sinkhorn
            :return:
                t: (B, N, D), the optimal transport matrix
        """
        assert f_method in ('badmm-e', 'badmm-q', 'sinkhorn')
        with torch.no_grad():
            if f_method == 'badmm-e':
                t = rot_badmm(x, c1, c2, p0, q0, a0, a1, a2, a3, mask, num, eps)
            elif f_method == 'badmm-q':
                t = rot_badmm2(x, c1, c2, p0, q0, a0, a1, a2, a3, mask, num, eps)
            else:
                t = rot_sinkhorn(x, c1, c2, p0, q0, a0, a1, a2, a3, mask, num, inner=5, eps=eps)
        ctx.save_for_backward(x, c1, c2, p0, q0, a0, a1, a2, a3, t)
        ctx.mask = mask
        ctx.num = num
        ctx.eps = eps
        ctx.f_method = f_method
        return t

    @staticmethod
    def backward(ctx, djdt):
        """Implement backward pass using implicit differentiation."""
        x, c1, c2, p0, q0, a0, a1, a2, a3, t = ctx.saved_tensors
        # initialize backward gradients (-v^T H^{-1} B with v = dJdP and B = I or B = -1/r or B = -1/c)
        djdx = 1.0 * a1 * t * djdt
        djdc1 = None if not ctx.needs_input_grad[1] else torch.zeros_like(c1)
        djdc2 = None if not ctx.needs_input_grad[2] else torch.zeros_like(c2)
        djdp0 = None if not ctx.needs_input_grad[3] else torch.zeros_like(p0)
        djdq0 = None if not ctx.needs_input_grad[4] else torch.zeros_like(q0)
        djda0 = None if not ctx.needs_input_grad[5] else torch.zeros_like(a0)
        djda1 = None if not ctx.needs_input_grad[6] else torch.zeros_like(a1)
        djda2 = None if not ctx.needs_input_grad[7] else torch.zeros_like(a2)
        djda3 = None if not ctx.needs_input_grad[8] else torch.zeros_like(a3)
        # return gradients (None for num, f_method)
        return djdx, djdc1, djdc2, djdp0, djdq0, djda0, djda1, djda2, djda3, None, None, None, None, None


class ApproxROT(nn.Module):
    """
    Neural network layer to implement regularized optimal transport.

    Parameters:
    -----------
    :param num: int, the number of iterations
    :param eps: float, default: 1.0e-8
        The epsilon avoiding numerical instability
    :param f_method: str, default: 'badmm-e'
        The feed-forward method, badmm-e, badmm-q or sinkhorn
    """

    def __init__(self, num: int = 4, eps: float = 1e-8, f_method: str = 'badmm-e'):
        super(ApproxROT, self).__init__()
        self.num = num
        self.eps = eps
        self.f_method = f_method

    def forward(self, x, c1, c2, p0, q0, a0, a1, a2, a3, mask):
        """
        Solving regularized OT problem
        """
        t = RegularizedOptimalTransport.apply(x, c1, c2, p0, q0, a0, a1, a2, a3, mask,
                                              self.num, self.eps, self.f_method)
        return t


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
            t = rot_sinkhorn(x, c1, c2, p0, q0, a0, a1, a2, a3, mask, self.num, inner=5, eps=self.eps)
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

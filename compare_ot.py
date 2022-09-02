import torch
from pooling.rotlayers import uot_sinkhorn as uot_sinkhorn1
from uot_layers import uot_sinkhorn as uot_sinkhorn0
from torch_geometric.utils import softmax, to_dense_batch

a = 1
k = 4
B = 2
D = 5
N1 = 10
N2 = 7
x0 = -torch.rand(N1+N2, D)
batch = torch.ones(N1+N2)
batch[:N1] = 0
batch = batch.type(torch.LongTensor)

p0 = torch.ones(B, 1, D) / D  # (B, 1, D)
log_p0 = torch.log(p0)
tmp = torch.zeros_like(x0[:, 0].unsqueeze(1))
q0 = softmax(tmp, batch)  # (N, 1)
log_q0 = torch.log(q0)  # (N, 1)
q0, _ = to_dense_batch(q0, batch)  # (B, Nmax, 1)
log_q0, _ = to_dense_batch(log_q0, batch)  # (B, Nmax, 1)
x1, mask = to_dense_batch(x0, batch)  # (B, Nmax, D)
mask = mask.unsqueeze(2)
print(x0.shape, x1.shape, mask.shape, p0.shape, q0.shape)

Ns = [N1, N2]
trans0 = torch.zeros_like(x1)
trans1 = torch.zeros_like(x1)
for b in range(B):
    tmp = x1[b, :Ns[b], :]
    print(tmp.shape)
    trans0[b, :Ns[b], :] = uot_sinkhorn0(x=tmp,
                                         log_u1=log_q0[b, :Ns[b], :],
                                         log_u2=log_p0[b, :, :],
                                         a1=a,
                                         a2=a,
                                         a3=a,
                                         k=k)

trans1 = uot_sinkhorn1(x=x1,
                       p0=p0,
                       q0=q0,
                       a1=a * torch.ones(k),
                       a2=a * torch.ones(k),
                       a3=a * torch.ones(k),
                       mask=mask,
                       num=k)

print(trans0)

print(trans1)

print(torch.dist(trans1, trans0))

from pooling import rotlayers
from pooling import utils
import os
import torch

B = 1
N = 10
D = 5
k = 500
a_max = 1e4 * torch.ones(k)
a_min = torch.zeros(k)
rho_max = 1e4 * torch.ones(k)
rho_min = torch.ones(k)
eps = 0
names = ['mean', 'max', 'att']

# synthetic data
X = torch.randn(B, N, D)
c1 = torch.bmm(X.permute(0, 2, 1), X) / X.shape[1]  # (B, D, D)
c2 = torch.bmm(X, X.permute(0, 2, 1)) / X.shape[2]  # (B, N, N)
mask = torch.ones(B, N, D)
q = torch.rand(N, 1)
q /= q.sum()
p0 = torch.ones(B, 1, D) / D
q0 = torch.ones(B, N, 1) / N
qa = torch.ones(B, 1, 1) * q.unsqueeze(0)

# ground truth of different poolings
pool_mean = torch.ones(N, D) / (N * D)
pool_max = torch.zeros(N, D)
pool_att = q @ torch.ones(1, D) / D
for n in range(N):
    idx = torch.argmax(X[0, n, :])
    pool_max[n, idx] = 1 / N
pools = [torch.t(pool_mean), torch.t(pool_max), torch.t(pool_att)]

# sinkhorn setting: p0, q0, a1, a2, a3, num, rho
set_mean = [p0, q0, a_max, a_max, a_max, k, rho_max]
set_max = [p0, q0, a_min, a_min, a_max, k, rho_min]
set_att = [p0, qa, a_max, a_max, a_max, k, rho_max]
sets = [set_mean, set_max, set_att]
trans1 = []
trans2 = []
trans3 = []


for i in range(3):
    print(names[i])
    transS = rotlayers.rot_sinkhorn(x=X,
                                    c1=c1,
                                    c2=c2,
                                    p0=sets[i][0],
                                    q0=sets[i][1],
                                    a0=torch.zeros(k),
                                    a1=sets[i][2],
                                    a2=sets[i][3],
                                    a3=sets[i][4],
                                    mask=mask,
                                    num=sets[i][5])

    transS /= transS.sum()
    trans1.append(torch.t(transS[0, :, :]))

    transE = rotlayers.rot_badmm(x=X,
                                 c1=c1,
                                 c2=c2,
                                 p0=sets[i][0],
                                 q0=sets[i][1],
                                 a0=torch.zeros(k),
                                 a1=sets[i][2],
                                 a2=sets[i][3],
                                 a3=sets[i][4],
                                 rho=sets[i][6],
                                 mask=mask,
                                 num=sets[i][5],
                                 eps=eps)
    transE /= transE.sum()
    trans2.append(torch.t(transE[0, :, :]))

    transQ = rotlayers.rot_badmm2(x=X,
                                  c1=c1,
                                  c2=c2,
                                  p0=sets[i][0],
                                  q0=sets[i][1],
                                  a0=torch.zeros(k),
                                  a1=sets[i][2],
                                  a2=sets[i][3],
                                  a3=sets[i][4],
                                  rho=sets[i][6],
                                  mask=mask,
                                  num=sets[i][5],
                                  eps=eps)
    transQ /= transQ.sum()
    trans3.append(torch.t(transQ[0, :, :]))

utils.visualize_data(x=torch.t(X[0, :, :]).numpy(),
                     dst=os.path.join('results', 'toy_data.pdf'))
for i in range(3):
    data_list = [pools[i].numpy(), trans1[i].numpy(), trans2[i].numpy(), trans3[i].numpy()]
    dst = os.path.join('results', 'cmp_{}.pdf'.format(names[i]))
    if i == 0:
        utils.visualize_pooling(data_list=data_list, dst=dst, vmin=0, vmax=0.04)
    else:
        utils.visualize_pooling(data_list=data_list, dst=dst)

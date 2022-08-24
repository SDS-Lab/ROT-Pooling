import os
from pooling import rotlayers
from pooling import utils
import torch
import numpy as np

a0 = 1
a1 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
a2 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
k = 500
eps = 0

B = 1
N = 10
D = 5
X = torch.randn(B, N, D)
c1 = torch.bmm(X.permute(0, 2, 1), X) / X.shape[1]  # (B, D, D)
c2 = torch.bmm(X, X.permute(0, 2, 1)) / X.shape[2]  # (B, N, N)
mask = torch.ones(B, N, D)
p0 = torch.ones(B, 1, D) / D
q0 = torch.ones(B, N, 1) / N

sinkhorn_map = np.zeros((len(a1), len(a2), 2))
badmm1_map = np.zeros((len(a1), len(a2), 2))
badmm2_map = np.zeros((len(a1), len(a2), 2))

for i in range(len(a1)):
    for j in range(len(a2)):
        transS = rotlayers.uot_sinkhorn(x=X,
                                        p0=p0,
                                        q0=q0,
                                        a1=torch.tensor(a1[i]),
                                        a2=torch.tensor(a2[j]),
                                        a3=torch.tensor(a2[j]),
                                        mask=mask,
                                        num=k)
        sinkhorn_map[i, j, 0] = transS.sum().numpy()

        transE = rotlayers.uot_badmm(x=X,
                                     p0=p0,
                                     q0=q0,
                                     a1=torch.tensor(a1[i]),
                                     a2=torch.tensor(a2[j]),
                                     a3=torch.tensor(a2[j]),
                                     mask=mask,
                                     num=k)
        badmm1_map[i, j, 0] = transE.sum().numpy()

        transQ = rotlayers.uot_badmm2(x=X,
                                      p0=p0,
                                      q0=q0,
                                      a1=torch.tensor(a1[i]),
                                      a2=torch.tensor(a2[j]),
                                      a3=torch.tensor(a2[j]),
                                      mask=mask,
                                      num=k)
        badmm2_map[i, j, 0] = transQ.sum().numpy()

        print(i, j, sinkhorn_map[i, j, 0], badmm1_map[i, j, 0], badmm2_map[i, j, 0])

        transS = rotlayers.rot_sinkhorn(x=X,
                                        c1=c1,
                                        c2=c2,
                                        p0=p0,
                                        q0=q0,
                                        a0=torch.tensor(a0),
                                        a1=torch.tensor(a1[i]),
                                        a2=torch.tensor(a2[j]),
                                        a3=torch.tensor(a2[j]),
                                        mask=mask,
                                        num=k)
        sinkhorn_map[i, j, 1] = transS.sum().numpy()

        transE = rotlayers.rot_badmm(x=X,
                                     c1=c1,
                                     c2=c2,
                                     p0=p0,
                                     q0=q0,
                                     a0=torch.tensor(a0),
                                     a1=torch.tensor(a1[i]),
                                     a2=torch.tensor(a2[j]),
                                     a3=torch.tensor(a2[j]),
                                     mask=mask,
                                     num=k)
        badmm1_map[i, j, 1] = transE.sum().numpy()

        transQ = rotlayers.rot_badmm2(x=X,
                                      c1=c1,
                                      c2=c2,
                                      p0=p0,
                                      q0=q0,
                                      a0=torch.tensor(a0),
                                      a1=torch.tensor(a1[i]),
                                      a2=torch.tensor(a2[j]),
                                      a3=torch.tensor(a2[j]),
                                      mask=mask,
                                      num=k)
        badmm2_map[i, j, 1] = transQ.sum().numpy()

        print(i, j, sinkhorn_map[i, j, 1], badmm1_map[i, j, 1], badmm2_map[i, j, 1])


sinkhorn_map[np.isinf(sinkhorn_map)] = 100
sinkhorn_map[sinkhorn_map > 99] = 99
badmm1_map[np.isinf(badmm1_map)] = 100
badmm2_map[np.isinf(badmm2_map)] = 100

astr = ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
xlabel = r'$\log_{10}\alpha_2$ and $\log_{10}\alpha_3$'
ylabel = r'$\log_{10}\alpha_1$'

result = os.path.join('results', 'map_uot_sinkhorn.pdf')
utils.visualize_map(data=sinkhorn_map[:, :, 0], dst=result, ticklabels=astr, xlabel=xlabel, ylabel=ylabel)
result = os.path.join('results', 'map_uot_badmme.pdf')
utils.visualize_map(data=badmm1_map[:, :, 0], dst=result, ticklabels=astr, xlabel=xlabel, ylabel=ylabel)
result = os.path.join('results', 'map_uot_badmmq.pdf')
utils.visualize_map(data=badmm2_map[:, :, 0], dst=result, ticklabels=astr, xlabel=xlabel, ylabel=ylabel)
result = os.path.join('results', 'map_rot_sinkhorn.pdf')
utils.visualize_map(data=sinkhorn_map[:, :, 1], dst=result, ticklabels=astr, xlabel=xlabel, ylabel=ylabel)
result = os.path.join('results', 'map_rot_badmme.pdf')
utils.visualize_map(data=badmm1_map[:, :, 1], dst=result, ticklabels=astr, xlabel=xlabel, ylabel=ylabel)
result = os.path.join('results', 'map_rot_badmmq.pdf')
utils.visualize_map(data=badmm2_map[:, :, 1], dst=result, ticklabels=astr, xlabel=xlabel, ylabel=ylabel)

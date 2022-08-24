from pooling import rotlayers
from pooling import utils
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 20})


Ns = [100, 200, 300, 400, 500]
Ks = [5, 10, 50, 100, 200, 400]
run = False

if run:
    B = 10
    n_trial = 50
    eps = 0
    D = 5
    a0 = torch.tensor(1)
    a1 = torch.tensor(1)
    a2 = torch.tensor(1)
    a3 = torch.tensor(1)
    runtime_N = np.zeros((len(Ns), n_trial, 3, 2))
    runtime_K = np.zeros((len(Ks), n_trial, 3, 2))

    for n in range(n_trial):
        print('Trial {}/{}'.format(n+1, n_trial))
        k = 100
        for i in range(len(Ns)):
            N = Ns[i]
            X = torch.randn(B, N, D)
            c1 = torch.bmm(X.permute(0, 2, 1), X) / X.shape[1]  # (B, D, D)
            c2 = torch.bmm(X, X.permute(0, 2, 1)) / X.shape[2]  # (B, N, N)
            mask = torch.ones(B, N, D)
            p0 = torch.ones(B, 1, D) / D
            q0 = torch.ones(B, N, 1) / N

            since = time.time()
            _ = rotlayers.rot_sinkhorn(x=X,
                                       c1=c1,
                                       c2=c2,
                                       p0=p0,
                                       q0=q0,
                                       a0=a0,
                                       a1=a1,
                                       a2=a2,
                                       a3=a3,
                                       mask=mask,
                                       num=k)
            runtime_N[i, n, 0, 0] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.rot_badmm(x=X,
                                    c1=c1,
                                    c2=c2,
                                    p0=p0,
                                    q0=q0,
                                    a0=a0,
                                    a1=a1,
                                    a2=a2,
                                    a3=a3,
                                    mask=mask,
                                    num=k)
            runtime_N[i, n, 1, 0] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.rot_badmm2(x=X,
                                     c1=c1,
                                     c2=c2,
                                     p0=p0,
                                     q0=q0,
                                     a0=a0,
                                     a1=a1,
                                     a2=a2,
                                     a3=a3,
                                     mask=mask,
                                     num=k)
            runtime_N[i, n, 2, 0] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.uot_sinkhorn(x=X,
                                       p0=p0,
                                       q0=q0,
                                       a1=a1,
                                       a2=a2,
                                       a3=a3,
                                       mask=mask,
                                       num=k)
            runtime_N[i, n, 0, 1] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.uot_badmm(x=X,
                                    p0=p0,
                                    q0=q0,
                                    a1=a1,
                                    a2=a2,
                                    a3=a3,
                                    mask=mask,
                                    num=k)
            runtime_N[i, n, 1, 1] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.uot_badmm2(x=X,
                                     p0=p0,
                                     q0=q0,
                                     a1=a1,
                                     a2=a2,
                                     a3=a3,
                                     mask=mask,
                                     num=k)
            runtime_N[i, n, 2, 1] = (time.time() - since) / B

        N = 100
        for i in range(len(Ks)):
            k = Ks[i]
            X = torch.randn(B, N, D)
            c1 = torch.bmm(X.permute(0, 2, 1), X) / X.shape[1]  # (B, D, D)
            c2 = torch.bmm(X, X.permute(0, 2, 1)) / X.shape[2]  # (B, N, N)
            mask = torch.ones(B, N, D)
            p0 = torch.ones(B, 1, D) / D
            q0 = torch.ones(B, N, 1) / N

            since = time.time()
            _ = rotlayers.rot_sinkhorn(x=X,
                                       c1=c1,
                                       c2=c2,
                                       p0=p0,
                                       q0=q0,
                                       a0=a0,
                                       a1=a1,
                                       a2=a2,
                                       a3=a3,
                                       mask=mask,
                                       num=k)
            runtime_K[i, n, 0, 0] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.rot_badmm(x=X,
                                    c1=c1,
                                    c2=c2,
                                    p0=p0,
                                    q0=q0,
                                    a0=a0,
                                    a1=a1,
                                    a2=a2,
                                    a3=a3,
                                    mask=mask,
                                    num=k)
            runtime_K[i, n, 1, 0] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.rot_badmm2(x=X,
                                     c1=c1,
                                     c2=c2,
                                     p0=p0,
                                     q0=q0,
                                     a0=a0,
                                     a1=a1,
                                     a2=a2,
                                     a3=a3,
                                     mask=mask,
                                     num=k)
            runtime_K[i, n, 2, 0] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.uot_sinkhorn(x=X,
                                       p0=p0,
                                       q0=q0,
                                       a1=a1,
                                       a2=a2,
                                       a3=a3,
                                       mask=mask,
                                       num=k)
            runtime_K[i, n, 0, 1] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.uot_badmm(x=X,
                                    p0=p0,
                                    q0=q0,
                                    a1=a1,
                                    a2=a2,
                                    a3=a3,
                                    mask=mask,
                                    num=k)
            runtime_K[i, n, 1, 1] = (time.time() - since) / B

            since = time.time()
            _ = rotlayers.uot_badmm2(x=X,
                                     p0=p0,
                                     q0=q0,
                                     a1=a1,
                                     a2=a2,
                                     a3=a3,
                                     mask=mask,
                                     num=k)
            runtime_K[i, n, 2, 1] = (time.time() - since) / B

    np.save(os.path.join('results', 'runtime_N.npy'), runtime_N)
    np.save(os.path.join('results', 'runtime_K.npy'), runtime_K)
else:
    runtime_N = np.load(os.path.join('results', 'runtime_N.npy'))
    runtime_K = np.load(os.path.join('results', 'runtime_K.npy'))


colors = ['red', 'blue']
names = ['Sinkhorn', 'BADMM-E']

tK_m = np.mean(runtime_K[:, :, :, 0], axis=1)
tK_v = np.std(runtime_K[:, :, :, 0], axis=1)
tN_m = np.mean(runtime_N[:, :, :, 0], axis=1)
tN_v = np.std(runtime_N[:, :, :, 0], axis=1)

utils.visualize_errorbar_curve(xs=Ks, ms=tK_m, vs=tK_v, colors=colors, labels=names,
                               xlabel='K', ylabel='Runtime (second)', dst=os.path.join('results', 'rot_KT.pdf'))
utils.visualize_errorbar_curve(xs=Ns, ms=tN_m, vs=tN_v, colors=colors, labels=names,
                               xlabel='N', ylabel='Runtime (second)', dst=os.path.join('results', 'rot_NT.pdf'))

tK_m = np.mean(runtime_K[:, :, :, 1], axis=1)
tK_v = np.std(runtime_K[:, :, :, 1], axis=1)
tN_m = np.mean(runtime_N[:, :, :, 1], axis=1)
tN_v = np.std(runtime_N[:, :, :, 1], axis=1)

utils.visualize_errorbar_curve(xs=Ks, ms=tK_m, vs=tK_v, colors=colors, labels=names,
                               xlabel='K', ylabel='Runtime (second)', dst=os.path.join('results', 'uot_KT.pdf'))
utils.visualize_errorbar_curve(xs=Ns, ms=tN_m, vs=tN_v, colors=colors, labels=names,
                               xlabel='N', ylabel='Runtime (second)', dst=os.path.join('results', 'uot_NT.pdf'))

# plt.figure(figsize=(5, 5))
# for i in range(2):
#     plt.plot(Ks, tK_m[:, i], 'o-', label=names[i], color=colors[i])
#     plt.fill_between(Ks, tK_m[:, i] - tK_v[:, i], tK_m[:, i] + tK_v[:, i],
#                      color=colors[i], alpha=0.2)
# plt.legend(loc='upper left')
# plt.xlabel('K')
# plt.ylabel('Runtime (second)')
# plt.grid()
# plt.tight_layout()
# plt.savefig('K_vs_t1.pdf')
# plt.close()
#
# plt.figure(figsize=(5, 5))
# for i in range(2):
#     plt.plot(Ns, tN_m[:, i], 'o-', label=names[i], color=colors[i])
#     plt.fill_between(Ns, tN_m[:, i] - tN_v[:, i], tN_m[:, i] + tN_v[:, i],
#                      color=colors[i], alpha=0.2)
# plt.legend(loc='upper left')
# plt.xlabel('N')
# plt.ylabel('Runtime (second)')
# plt.grid()
# plt.tight_layout()
# plt.savefig('N_vs_t1.pdf')
# plt.close()

# plt.figure(figsize=(5, 5))
# for i in range(2):
#     plt.plot(Ks, tK_m[:, i], 'o-', label=names[i], color=colors[i])
#     plt.fill_between(Ks, tK_m[:, i] - tK_v[:, i], tK_m[:, i] + tK_v[:, i],
#                      color=colors[i], alpha=0.2)
# plt.legend(loc='upper left')
# plt.xlabel('K')
# plt.ylabel('Runtime (second)')
# plt.grid()
# plt.tight_layout()
# plt.savefig('K_vs_t2.pdf')
# plt.close()
#
# plt.figure(figsize=(5, 5))
# for i in range(2):
#     plt.plot(Ns, tN_m[:, i], 'o-', label=names[i], color=colors[i])
#     plt.fill_between(Ns, tN_m[:, i] - tN_v[:, i], tN_m[:, i] + tN_v[:, i],
#                      color=colors[i], alpha=0.2)
# plt.legend(loc='upper left')
# plt.xlabel('N')
# plt.ylabel('Runtime (second)')
# plt.grid()
# plt.tight_layout()
# plt.savefig('N_vs_t2.pdf')
# plt.close()

from pooling import utils
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pooling as Pooling
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 20})
poolings = ['add_pooling',
            'mean_pooling',
            'max_pooling',
            'deepset',
            'mix_pooling',
            'gated_pooling',
            'set_set',
            'attention_pooling',
            'gated_attention_pooling',
            'dynamic_pooling',
            'GeneralizedNormPooling',
            # 'SAGPooling',
            # 'ASAPooling',
            'uot_pooling_sinkhorn',
            'uot_pooling_badmm-e',
            'uot_pooling_badmm-q']

run = False
num = 4
Dim = [10,50,100,250,500]
trial = 10
batch_size =10
if run:
    sampleN = 500
    batch =[]
    for i in range(batch_size):
        for j in range(sampleN):
            batch.append(i)
    batch = torch.tensor(batch)
    runtime_dim = np.zeros((len(Dim),len(poolings), trial))
    for i in range(len(Dim)):
        X = torch.randn(batch_size*sampleN, Dim[i])
        for pooling in range(len(poolings)):
            for n in range(trial):
                print('Dim{}-Pooling{}-Trial {}/{}'.format(Dim[i], poolings[pooling], n + 1, trial))
                if poolings[pooling] == 'add_pooling':
                    since = time.time()
                    _ = global_add_pool(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'mean_pooling':
                    since = time.time()
                    _ = global_mean_pool(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'max_pooling':
                    since = time.time()
                    _ = global_max_pool(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'deepset':
                    pooling_tmp = Pooling.DeepSet(Dim[i], 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'mix_pooling':
                    pooling_tmp = Pooling.MixedPooling()
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'gated_pooling':
                    pooling_tmp = Pooling.GatedPooling(Dim[i])
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'set_set':
                    torch.backends.cudnn.enabled = False
                    pooling_tmp = Pooling.Set2Set(Dim[i], 2, 1)
                    dense_tmp = torch.nn.Linear(Dim[i] * 2, 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    _ = dense_tmp(_)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'attention_pooling':
                    pooling_tmp = Pooling.AttentionPooling(Dim[i], 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'gated_attention_pooling':
                    pooling_tmp = Pooling.GatedAttentionPooling(Dim[i], 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'dynamic_pooling':
                    pooling_tmp = Pooling.DynamicPooling(Dim[i], 3)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'GeneralizedNormPooling':
                    pooling_tmp = Pooling.GeneralizedNormPooling(Dim[i])
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                # if poolings[pooling] == 'SAGPooling':
                #     pooling_tmp = SAGPooling(Dim[i])
                #     since = time.time()
                #     _ = pooling_tmp(X, batch)
                #     runtime_dim[i, pooling, n] = time.time() - since
                # if poolings[pooling] == 'ASAPooling':
                #     pooling_tmp = ASAPooling(Dim[i])
                #     since = time.time()
                #     _ = pooling_tmp(X, batch)
                #     runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'uot_pooling_sinkhorn':
                    pooling_tmp = Pooling.UOTPooling(dim=Dim[i], num=num, f_method='sinkhorn')
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'uot_pooling_badmm-e':
                    pooling_tmp = Pooling.UOTPooling(dim=Dim[i], num=num, f_method='badmm-e')
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'uot_pooling_badmm-q':
                    pooling_tmp = Pooling.UOTPooling(dim=Dim[i], num=num, f_method='badmm-q')
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
    np.save(os.path.join('results', 'num_' + str(num) + '_poolings_runtime_dim.npy'), runtime_dim)


    #samples
    dim = 50
    sampleN = [10, 50, 100, 250, 500]
    runtime_dim = np.zeros((len(sampleN), len(poolings), trial))
    for i in range(len(sampleN)):
        batch = []
        for m in range(batch_size):
            for n in range(sampleN[i]):
                batch.append(m)
        batch = torch.tensor(batch)
        X = torch.randn(batch_size * sampleN[i], dim)
        for pooling in range(len(poolings)):
            for n in range(trial):
                print('Sample{}-Pooling{}-Trial {}/{}'.format(Dim[i], poolings[pooling], n + 1, trial))
                if poolings[pooling] == 'add_pooling':
                    since = time.time()
                    _ = global_add_pool(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'mean_pooling':
                    since = time.time()
                    _ = global_mean_pool(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'max_pooling':
                    since = time.time()
                    _ = global_max_pool(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'deepset':
                    pooling_tmp = Pooling.DeepSet(dim, 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'mix_pooling':
                    pooling_tmp = Pooling.MixedPooling()
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'gated_pooling':
                    pooling_tmp = Pooling.GatedPooling(dim)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'set_set':
                    torch.backends.cudnn.enabled = False
                    pooling_tmp = Pooling.Set2Set(dim, 2, 1)
                    dense_tmp = torch.nn.Linear(dim * 2, 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    _ = dense_tmp(_)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'attention_pooling':
                    pooling_tmp = Pooling.AttentionPooling(dim, 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'gated_attention_pooling':
                    pooling_tmp = Pooling.GatedAttentionPooling(dim, 32)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'dynamic_pooling':
                    pooling_tmp = Pooling.DynamicPooling(dim, 3)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'GeneralizedNormPooling':
                    pooling_tmp = Pooling.GeneralizedNormPooling(dim)
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                # if poolings[pooling] == 'SAGPooling':
                #     pooling_tmp = SAGPooling(Dim[i])
                #     since = time.time()
                #     _ = pooling_tmp(X, batch)
                #     runtime_dim[i, pooling, n] = time.time() - since
                # if poolings[pooling] == 'ASAPooling':
                #     pooling_tmp = ASAPooling(Dim[i])
                #     since = time.time()
                #     _ = pooling_tmp(X, batch)
                #     runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'uot_pooling_sinkhorn':
                    pooling_tmp = Pooling.UOTPooling(dim=dim, num=num, f_method='sinkhorn')
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'uot_pooling_badmm-e':
                    pooling_tmp = Pooling.UOTPooling(dim=dim, num=num, f_method='badmm-e')
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
                if poolings[pooling] == 'uot_pooling_badmm-q':
                    pooling_tmp = Pooling.UOTPooling(dim=dim, num=num, f_method='badmm-q')
                    since = time.time()
                    _ = pooling_tmp(X, batch)
                    runtime_dim[i, pooling, n] = time.time() - since
    np.save(os.path.join('results', 'num_' + str(num) + '_poolings_runtime_sample.npy'), runtime_dim)
else:
    #dims
    num = 4
    runtime_dim = np.load(os.path.join('results', 'num_' + str(num) + '_poolings_runtime_dim.npy'))
    runtime_dim = runtime_dim[:, 10:14, :]
    runtime_dim_mean = np.mean(runtime_dim, axis=2)
    runtime_dim_std = np.std(runtime_dim, axis=2)
    print("end")
    colors = ['red', 'blue', 'green', 'black']
    names = poolings[10:14]
    utils.visualize_errorbar_curve(xs=Dim, ms=runtime_dim_mean, vs=runtime_dim_std, colors=colors, labels=names,
                                   xlabel='Dim', ylabel='Runtime (second)',
                                   dst=os.path.join('results', 'num_' + str(num) + '_runtime_dim.pdf'))

    #samples
    runtime_sample = np.load(os.path.join('results', 'num_' + str(num) + '_poolings_runtime_sample.npy'))
    runtime_sample = runtime_sample[:, 10:14, :]
    runtime_sample_mean = np.mean(runtime_sample, axis=2)
    runtime_sample_std = np.std(runtime_sample, axis=2)
    print("end")
    colors = ['red', 'blue', 'green', 'black']
    names = poolings[10:14]
    utils.visualize_errorbar_curve(xs=Dim, ms=runtime_sample_mean, vs=runtime_sample_std, colors=colors, labels=names,
                                   xlabel='Dim', ylabel='Runtime (second)',
                                   dst=os.path.join('results', 'num_' + str(num) + '_runtime_sample.pdf'))
import torch
import time
from pooling import rotpooling
import torch.optim as optim


dim = 20
num = 4
bs = 50
ns = 10
eps = 1e-8
torch.manual_seed(10)
f_methods = ['badmm-e', 'badmm-q', 'sinkhorn']
pooling_types = ['uot', 'rot']
x = torch.randn(bs * ns, dim)
batch = torch.zeros(bs)
for i in range(ns - 1):
    batch = torch.concat((batch, (i+1) * torch.ones(bs)), dim=0)
batch = batch.type(torch.LongTensor)


for f in f_methods:
    for b in pooling_types:
        if b == 'rot':
            model = rotpooling.ROTPooling(dim=dim, num=num, eps=eps, f_method=f)
            # print(model.a0, model.a1, model.a2, model.a3)
        else:
            model = rotpooling.UOTPooling(dim=dim, num=num, eps=eps, f_method=f)
            # print(model.a1, model.a2, model.a3)
        optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.05, momentum=0.9)

        model.train()
        optimizer.zero_grad()

        since = time.time()
        out = model(x, batch)
        loss = torch.sum(out ** 2)
        loss.backward()
        optimizer.step()
        stop = time.time()
        # if b == 'rot':
        #     print(model.a0, model.a1, model.a2, model.a3)
        # else:
        #     print(model.a1, model.a2, model.a3)
        print('{}, {}: runtime={:.6f}sec\n'.format(f, b, stop - since))

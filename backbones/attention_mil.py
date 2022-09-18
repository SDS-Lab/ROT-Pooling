import argparse
import time

import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.loader import DataLoader

import random
import os
import sys
sys.path.append("..")
import pooling as Pooling


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DS",
        help="dataset to train on, like musk1 or fox",
        default="datasets/Biocreative/component",
        type=str,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        metavar="LR",
        help="learning rate (default: 0.0005)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.005,
        help="weight_decay (default: 0.005)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size (default: 128)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    # for uot-pooling
    parser.add_argument("--a1", type=float, default=None)
    parser.add_argument("--a2", type=float, default=None)
    parser.add_argument("--a3", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--u1", type=str, default="fixed")
    parser.add_argument("--u2", type=str, default="fixed")
    parser.add_argument("--pooling_layer", help="score_pooling", default="deepset", type=str)
    parser.add_argument("--f_method", type=str, default="badmm-e")
    return parser.parse_args()


def load_mil_data_mat(
    dataset, n_folds, batch_size: int, normalize: bool = True, split: float = 0.75, seed: int = 1):
    data = sio.loadmat(dataset + ".mat")
    instances = data["data"]
    bags = []
    labels = []
    for i in range(len(instances)):
        bag = torch.from_numpy(instances[i][0]).float()[:, 0:-1]
        label = instances[i][1][0, 0]
        bags.append(bag)
        if label < 0:
            labels.append(0)
        else:
            labels.append(label)
    labels = torch.Tensor(labels).long()

    if normalize:
        all_instances = torch.cat(bags, dim=0)
        avg_instance = torch.mean(all_instances, dim=0, keepdim=True)
        std_instance = torch.std(all_instances, dim=0, keepdim=True)
        for i in range(len(bags)):
            bags[i] = (bags[i] - avg_instance) / (std_instance + 1e-6)
    bags = bags
    bags_fea = []
    for i in range(len(bags)):
        bags_fea.append(Data(x=bags[i], y=labels[i]))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
    dataloaders = []
    for train_idx, test_idx in kf.split(bags_fea):
        dataloader = {}
        dataloader["train"] = DataLoader([bags_fea[ibag] for ibag in train_idx], batch_size=batch_size, shuffle=True)
        dataloader["test"] = DataLoader([bags_fea[ibag] for ibag in test_idx], batch_size=batch_size, shuffle=False)
        dataloaders.append(dataloader)
    return dataloaders


def get_dim(dataset):
    data = sio.loadmat(dataset + ".mat")
    ins_fea = data["data"][0, 0]
    length = len(ins_fea[0]) - 1
    return length


class Net(nn.Module):
    def __init__(self, dim, pooling_layer, a1, a2, a3, rho, u1, u2, k, h):
        super(Net, self).__init__()
        self.dim = dim
        self.pooling_layer = pooling_layer
        self.L = 64
        self.D = 64
        self.K = 1
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.rho = rho
        self.u1 = u1
        self.u2 = u2
        self.k = k
        self.h = h
        self.rho = rho
        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
        )
        if self.pooling_layer == "mix_pooling":
            self.pooling = Pooling.MixedPooling()
        if self.pooling_layer == "gated_pooling":
            self.pooling = Pooling.GatedPooling(64)
        if self.pooling_layer == "set_set":
            self.pooling = Pooling.Set2Set(64, 2, 1)
            self.dense4 = torch.nn.Linear(64 * 2, 64)
        if self.pooling_layer == "attention_pooling":
            self.pooling = Pooling.AttentionPooling(64, 64)
        if self.pooling_layer == "gated_attention_pooling":
            self.pooling = Pooling.GatedAttentionPooling(64, 64)
        if self.pooling_layer == "dynamic_pooling":
            self.pooling = Pooling.DynamicPooling(64, 3)
        if self.pooling_layer == 'GeneralizedNormPooling':
            self.pooling = Pooling.GeneralizedNormPooling(d=64)
        if self.pooling_layer == "uot_pooling":
            self.pooling = Pooling.UOTPooling(dim=64, num=k, a1=a1, a2=a2, a3=a3, p0=u1, q0=u2, f_method=args.f_method)
        if self.pooling_layer == "rot_pooling":
            self.pooling = Pooling.ROTPooling(dim=64, num=k, a1=a1, a2=a2, a3=a3, p0=u1, q0=u2, f_method=args.f_method)
        if self.pooling_layer == "deepset":
            self.pooling = Pooling.DeepSet(64, 64)
        self.classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x, batch):
        H = self.feature_extractor_part1(x)
        if self.pooling_layer == "add_pooling":
            M = global_add_pool(H, batch)
        elif self.pooling_layer == "mean_pooling":
            M = global_mean_pool(H, batch)
        elif self.pooling_layer == "max_pooling":
            M = global_max_pool(H, batch)
        # for set_set
        elif self.pooling_layer == "set_set":
            torch.backends.cudnn.enabled = False
            M = self.pooling(H, batch)
            M = self.dense4(M)
        else:
            M = self.pooling(H, batch)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat

    def calculate_classification_error(self, X, batch, Y):
        Y = Y.float()
        _, Y_hat = self.forward(X, batch)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        acc_num = Y_hat.eq(Y).cpu().float().sum().item()
        return error, Y_hat, acc_num

    def calculate_objective(self, X, batch, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X, batch)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli
        return neg_log_likelihood


def train(model, optimizer, train_bags):
    model.train()
    train_loss = 0.0
    train_error = 0.0
    for idx, data_a in enumerate(train_bags):
        data = data_a.x
        bag_label = data_a.y
        batch = data_a.batch
        if args.cuda:
            data, bag_label, batch = data.cuda(), bag_label.cuda(), batch.cuda()
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, batch, bag_label)
        train_loss += loss.item()
        error, _, acc_num = model.calculate_classification_error(data, batch, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_bags)
    train_error /= len(train_bags)
    return train_error, train_loss


def acc_test(model, test_bags):
    model.eval()
    test_loss = 0.0
    test_error = 0.0
    num_bags = 0
    num_corrects = 0
    for batch_idx, data_a in enumerate(test_bags):
        data = data_a.x
        bag_label = data_a.y
        batch = data_a.batch
        num_bags += bag_label.shape[0]
        if args.cuda:
            data, bag_label, batch = data.cuda(), bag_label.cuda(), batch.cuda()
        with torch.no_grad():
            loss, attention_weights = model.calculate_objective(data, batch, bag_label)
            test_loss += loss.item()
            error, predicted_label, acc_num = model.calculate_classification_error(data, batch, bag_label)
            test_error += error
            num_corrects += acc_num

    test_error /= len(test_bags)
    test_loss /= len(test_bags)
    test_acc = num_corrects / num_bags
    return test_error, test_loss, test_acc


def mil_Net(dataloader):
    dim = get_dim(args.DS)
    pooling_layer = args.pooling_layer
    a1 = args.a1
    a2 = args.a2
    a3 = args.a3
    rho = args.rho
    u1 = args.u1
    u2 = args.u2
    k = args.k
    h = args.h
    rho = rho
    print("Init Model")
    model = Net(dim, pooling_layer, a1, a2, a3, rho, u1, u2, k, h)
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
    )
    train_bags = dataloader["train"]
    test_bags = dataloader["test"]
    print(len(train_bags))
    print(len(test_bags))
    t1 = time.time()
    test_accs = []
    test_loss_error = []
    for epoch in range(1, args.epochs + 1):
        train_error, train_loss = train(model, optimizer, train_bags)
        test_error, test_loss, test_acc = acc_test(model, test_bags)
        test_accs.append(test_acc)
        test_loss_error.append(test_loss + test_error)
        print(
            "epoch=",
            epoch,
            "  train_error= {:.3f}".format(train_error),
            "  train_loss= {:.3f}".format(train_loss),
            "  test_error= {:.3f}".format(test_error),
            "  test_loss={:.3f}".format(test_loss),
            "  test_acc= {:.3f}".format(test_acc),
        )
    t2 = time.time()
    print("run time:", (t2 - t1) / 60.0, "min")
    index_epoch = np.argmin(test_loss_error)
    print("test_acc={:.3f}".format(test_accs[index_epoch]))
    return test_accs[index_epoch]


if __name__ == "__main__":
    args = arg_parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("\nGPU is ON!")
    print("Load Train and Test Set")
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    log_dir = (args.DS).split("/")[-1]

    run = 1
    n_folds = 5
    seed = [0]
    acc = np.zeros((run, n_folds), dtype=float)
    for irun in range(run):
        os.environ['PYTHONHASHSEED'] = str(seed[irun])
        torch.manual_seed(seed[irun])
        torch.cuda.manual_seed(seed[irun])
        torch.cuda.manual_seed_all(seed[irun])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        np.random.seed(seed[irun])
        random.seed(seed[irun])

        dataloaders = load_mil_data_mat(dataset=args.DS, n_folds=n_folds, normalize=True, batch_size=args.batch_size)
        for ifold in range(n_folds):
            print("run=", irun, "  fold=", ifold)
            acc[irun][ifold] = mil_Net(dataloaders[ifold])
    print("mi-net mean accuracy = ", np.mean(acc))
    print("std = ", np.std(acc))
    with open('logs/log_' + log_dir, 'a+') as f:
        paras_UOTpoolong = '[' + str(args.epochs) + "-" + str(args.k) + "-" + str(args.h) + "-" + str(args.a1) + "-" + str(args.a2) + "-" + str(
            args.a3) + "-" + str(args.u1) + "-" + str(args.u2) + "-" + str(args.rho) + ']'
        f.write('{}\n'.format(args.DS))
        f.write('{}\n'.format(args.pooling_layer))
        f.write('{}\n'.format(args.f_method))
        f.write('{}\n'.format(paras_UOTpoolong))
        f.write('{}\n'.format(np.mean(acc)))
        f.write('{}\n'.format(np.std(acc)))

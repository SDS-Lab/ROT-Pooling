import torch.nn.functional as F
import random
import util
import numpy as np
from torch_geometric.data import DataLoader
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    SAGPooling,
    ASAPooling,
)
from sklearn.metrics import roc_auc_score
from dataprocess_fears import *
import argparse
import os
import warnings
import logging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")
import pooling as Pooling


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--epoch", type=int, default=40, help="epoch")
    parser.add_argument("--dataset", type=str, default="fears", help="dataset")
    # for uot-pooling
    parser.add_argument("--a1", type=float, default=None)
    parser.add_argument("--a2", type=float, default=None)
    parser.add_argument("--a3", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--same_para", type=bool, default=False)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--p0", type=str, default="fixed")
    parser.add_argument("--q0", type=str, default="fixed")
    parser.add_argument("--f_method", type=str, default="sinkhorn")
    parser.add_argument("--eps", type=float, default=1e-18)
    parser.add_argument(
        "--pooling_layer", help="pooling_layer", default="uot_pooling", type=str
    )
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True)

class Net(torch.nn.Module):
    def __init__(
        self,
        pooling_layer,
        a1,
        a2,
        a3,
        rho,
        p0,
        q0,
        num,
        eps,
        same_para,
        f_method,
    ):
        super(Net, self).__init__()
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.f_method = f_method
        self.rho = rho
        self.same_para = same_para
        self.p0 = p0
        self.q0 = q0
        self.num = num
        self.eps = eps
        self.rho = rho
        self.pooling_layer = pooling_layer
        dim_h = 64
        #gin
        self.conv1 = GINConv(
            Sequential(Linear(30, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.lin1 = torch.nn.Linear(64, 128)
        self.lin2 = torch.nn.Linear(128, 1)

        # for pooling1
        feature_pooling = 64
        if self.pooling_layer == "mix_pooling":
            self.pooling = Pooling.MixedPooling()
        if self.pooling_layer == "gated_pooling":
            self.pooling = Pooling.GatedPooling(feature_pooling)
        if self.pooling_layer == "set_set":
            self.pooling = Pooling.Set2Set(feature_pooling, 2, 1)
            self.dense = torch.nn.Linear(feature_pooling * 2, 64)
        if self.pooling_layer == "attention_pooling":
            self.pooling = Pooling.AttentionPooling(feature_pooling, 32)
        if self.pooling_layer == "gated_attention_pooling":
            self.pooling = Pooling.GatedAttentionPooling(feature_pooling, 32)
        if self.pooling_layer == "dynamic_pooling":
            self.pooling = Pooling.DynamicPooling(feature_pooling, 3)
        if self.pooling_layer == "uot_pooling":
            self.pooling = Pooling.UOTPooling(
                dim=feature_pooling,
                num=num,
                rho=rho,
                same_para=same_para,
                p0=p0,
                q0=q0,
                eps=eps,
                a1=a1,
                a2=a2,
                a3=a3,
                f_method=f_method,
            )
        if self.pooling_layer == "rot_pooling":
            self.pooling = Pooling.ROTPooling(
                dim=feature_pooling,
                num=num,
                rho=rho,
                same_para=same_para,
                p0=p0,
                q0=q0,
                eps=eps,
                a1=a1,
                a2=a2,
                a3=a3,
                f_method=f_method,
            )
        if self.pooling_layer == "deepset":
            self.pooling = Pooling.DeepSet(feature_pooling, 32)
        if self.pooling_layer == "GeneralizedNormPooling":
            self.pooling = Pooling.GeneralizedNormPooling(feature_pooling)
        if self.pooling_layer == "SAGPooling":
            self.pooling = SAGPooling(feature_pooling)
        if self.pooling_layer == "ASAPooling":
            self.pooling = ASAPooling(feature_pooling)

        # for pooling2
        feature_pooling2 = 128
        if self.pooling_layer == "mix_pooling":
            self.pooling2 = Pooling.MixedPooling()
        if self.pooling_layer == "gated_pooling":
            self.pooling2 = Pooling.GatedPooling(feature_pooling2)
        if self.pooling_layer == "set_set":
            self.pooling2 = Pooling.Set2Set(feature_pooling2, 2, 1)
            self.dense2 = torch.nn.Linear(feature_pooling2 * 2, 128)
        if self.pooling_layer == "attention_pooling":
            self.pooling2 = Pooling.AttentionPooling(feature_pooling2, 32)
        if self.pooling_layer == "gated_attention_pooling":
            self.pooling2 = Pooling.GatedAttentionPooling(feature_pooling2, 32)
        if self.pooling_layer == "dynamic_pooling":
            self.pooling2 = Pooling.DynamicPooling(feature_pooling2, 3)
        if self.pooling_layer == "uot_pooling":
            self.pooling2 = Pooling.UOTPooling(
                dim=feature_pooling,
                num=num,
                rho=rho,
                same_para=same_para,
                p0=p0,
                q0=q0,
                eps=eps,
                a1=a1,
                a2=a2,
                a3=a3,
                f_method=f_method,
            )
        if self.pooling_layer == "rot_pooling":
            self.pooling2 = Pooling.ROTPooling(
                dim=feature_pooling,
                num=num,
                rho=rho,
                same_para=same_para,
                p0=p0,
                q0=q0,
                eps=eps,
                a1=a1,
                a2=a2,
                a3=a3,
                f_method=args.f_method,
            )
        if self.pooling_layer == "deepset":
            self.pooling2 = Pooling.DeepSet(feature_pooling2, 32)
        if self.pooling_layer == "GeneralizedNormPooling":
            self.pooling2 = Pooling.GeneralizedNormPooling(feature_pooling2)
        if self.pooling_layer == "SAGPooling":
            self.pooling2 = SAGPooling(feature_pooling2)
        if self.pooling_layer == "ASAPooling":
            self.pooling2 = ASAPooling(feature_pooling2)

    def forward(self, data, device):
        data = data.to(device)
        x, edge_index, pos_raw, batch, num = (
            data.x,
            data.edge_index,
            data.pos,
            data.batch,
            data.num,
        )
        nodes_orders, batch_orders = org_batch(num)
        nodes_orders = torch.tensor(nodes_orders).to(device)
        batch_orders = torch.tensor(batch_orders).to(device)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        # pooling1
        if self.pooling_layer == "add_pooling":
            x = global_add_pool(x, nodes_orders)
        elif self.pooling_layer == "mean_pooling":
            x = global_mean_pool(x, nodes_orders)
        elif self.pooling_layer == "max_pooling":
            x = global_max_pool(x, nodes_orders)
        # for set_set
        elif self.pooling_layer == "set_set":
            torch.backends.cudnn.enabled = False
            x = self.pooling(x, nodes_orders)
            x = self.dense(x)
        elif self.pooling_layer == 'SAGPooling':
            x, _, _, batch_graph, _, _ = self.pooling(x, edge_index, batch=nodes_orders)
            x = global_add_pool(x, batch_graph)
        elif self.pooling_layer == 'ASAPooling':
            x, _, _, batch_graph, _ = self.pooling(x, edge_index, batch=nodes_orders)
            x = global_add_pool(x, batch_graph)
        #
        else:
            x = self.pooling(x, nodes_orders)

        # pooling2
        x_len = 128
        x = self.lin1(x)
        x = F.relu(x)
        if self.pooling_layer == "add_pooling":
            x = global_add_pool(x, batch_orders)
        elif self.pooling_layer == "mean_pooling":
            x = global_mean_pool(x, batch_orders)
        elif self.pooling_layer == "max_pooling":
            x = global_max_pool(x, batch_orders)
        # for set_set
        elif self.pooling_layer == "set_set":
            torch.backends.cudnn.enabled = False
            x = self.pooling2(x, batch_orders)
            x = self.dense2(x)
        elif self.pooling_layer == 'SAGPooling':
            # pooling2
            x = global_add_pool(x, batch_orders)
        elif self.pooling_layer == 'ASAPooling':
            x = global_add_pool(x, batch_orders)
        #
        else:
            x = self.pooling2(x, batch_orders)
        x = torch.sigmoid(self.lin2(x)).squeeze(1)
        return x


def org_batch(num):
    """
    # generate nodes_orders and
    :param num: [[2,1][3][4,2,3]],[2,1] means a combination which has two graphs, the nodes of those graphs are 2 and 1.
    :return:
        nodes_orders: [0,0,1,2,2,2,3,3,3,3,4,4,5,5,5] responses to the nodes in the graphs of drug combinations.
        batch_orders: [0,0,1,2,2,2,3,3] responses to the graphs in drug combinations.
    """

    nodes_order = []
    batch_order = []
    for i in range(len(num)):
        batch_order.append(len(num[i]))
        for j in range(len(num[i])):
            nodes_order.append(num[i][j])
    nodes_orders = []
    batch_orders = []
    num = 0
    for i in range(len(batch_order)):
        for j in range(batch_order[i]):
            batch_orders.append(num)
        num += 1
    num_2 = 0
    for i in range(len(nodes_order)):
        for j in range(nodes_order[i]):
            nodes_orders.append(num_2)
        num_2 += 1
    return nodes_orders, batch_orders


def train(model, optimizer, crit, train_loader, train_dataset, device):
    model.train()
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        data_model = data
        output = model(data_model, device).squeeze()
        label = data.y.float().squeeze().to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data_model.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate(model, crit, loader, dataset, device):
    model.eval()
    predictions = []
    labels = []
    loss_all = 0
    with torch.no_grad():
        for data in loader:
            data_model = data
            pred = model(data_model, device).squeeze()
            label = data.y.float().squeeze().to(device)
            loss = crit(pred, label)
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            # print(pred)
            predictions.append(pred)
            labels.append(label)
            loss_all += data_model.num_graphs * loss.item()
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    val_loss = loss_all / len(dataset)
    predictions_tensor = torch.from_numpy(predictions)
    predictions_tensor = (predictions_tensor > 0.5).float()
    labels_tensor = torch.from_numpy(labels)
    acc, pre, rec, F1, auc = util.evaluation(predictions_tensor, labels_tensor)
    return val_loss, roc_auc_score(labels, predictions), acc, pre, rec, F1, auc


def run(args,seed):
    print("begin")
    setup_seed(seed)
    dataset = FearsGraphDataset(root="fears", name="fears")
    dataset = dataset.data
    random.shuffle(dataset)
    split_len = len(dataset)
    train_set_len = int(split_len * 0.6)
    valid_set_len = int(split_len * 0.2)
    train_dataset = dataset[:train_set_len]
    val_dataset = dataset[train_set_len : valid_set_len + train_set_len]
    test_dataset = dataset[valid_set_len + train_set_len :]
    print(len(dataset))
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("Data processing completed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(
        pooling_layer=args.pooling_layer,
        a1=args.a1,
        a2=args.a2,
        a3=args.a3,
        rho=args.rho,
        p0=args.p0,
        q0=args.q0,
        num=args.num,
        eps=args.eps,
        same_para=args.same_para,
        f_method=args.f_method,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.BCELoss()
    val_acc_all = []
    test_metric_all = []

    for epoch in range(args.epoch):
        val_val_loss, val_acc, val_acc2, val_pre, val_rec, val_F1, val_auc = evaluate(
            model, crit, val_loader, val_dataset, device
        )
        (test_val_loss,
            test_acc,
            test_acc2,
            test_pre,
            test_rec,
            test_F1,
            test_auc,
        ) = evaluate(model, crit, test_loader, test_dataset, device)
        val_acc_all.append(val_acc2)
        test_metric_all.append(
            [test_acc, test_acc2, test_pre, test_rec, test_F1, test_auc]
        )

    best_val_epoch = np.argmax(np.array(val_acc_all))
    best_test_result = test_metric_all[best_val_epoch]
    return best_test_result


if __name__ == "__main__":

    args = arg_parse()
    results=[]
    for i in [0,1,2,3,4]:
        results.append(run(args,i))
    result = np.mean(np.array(results), axis=0)
    std = np.std(np.array(results), axis=0)
    print(result)
    print(std)

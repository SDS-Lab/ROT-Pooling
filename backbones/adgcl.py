import logging
import random
import torch
import json
from sklearn.svm import LinearSVC, SVC
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
import os
import os.path as osp
import shutil
from sklearn.metrics import accuracy_score
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
import argparse
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import numpy as np
import torch.nn.functional as F
from typing import Callable, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, SAGPooling, ASAPooling
import sys
sys.path.append("..")
import pooling as Pooling
import warnings
warnings.filterwarnings("ignore")
import time

def arg_parse():
    parser = argparse.ArgumentParser(description="AD-GCL TU")
    # MUTAG  DD PROTEINS NCI1
    # COLLAB REDDIT-BINARY REDDIT-MULTI-5K IMDB-BINARY,IMDB-MULTI
    # NCI109 PTC_MR
    parser.add_argument("--DS", type=str, default="MUTAG", help="Dataset")
    parser.add_argument("--model_lr", type=float, default=0.001, help="Model Learning rate.")
    parser.add_argument("--view_lr", type=float, default=0.001, help="View Learning rate.")
    parser.add_argument("--num_gc_layers", type=int, default=5, help="Number of GNN layers before pooling",)
    parser.add_argument("--pooling_type", type=str, default="standard", help="GNN Pooling Type Standard/Layerwise",)
    parser.add_argument("--emb_dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--mlp_edge_model_dim", type=int, default=64, help="embedding dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--drop_ratio", type=float, default=0.5, help="Dropout Ratio / Probability")
    parser.add_argument("--epochs", type=int, default=20, help="Train Epochs")
    parser.add_argument("--reg_lambda", type=float, default=5.0, help="View Learner Edge Perturb Regularization Strength",)
    parser.add_argument("--eval_interval", type=int, default=1, help="eval epochs interval")
    parser.add_argument("--downstream_classifier",type=str,default="linear",help="Downstream classifier is linear or non-linear",)
    parser.add_argument("--seed", type=int, default=0)
    # for uot-pooling
    parser.add_argument("--a1", type=float, default=None)
    parser.add_argument("--a2", type=float, default=None)
    parser.add_argument("--a3", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--same_para", type=bool, default=False)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument('--p0', type=str, default='fixed')
    parser.add_argument('--q0', type=str, default='fixed')
    parser.add_argument("--f_method", type=str, default="badmm-e")
    parser.add_argument("--eps", type=float, default=1e-18)
    parser.add_argument("--pooling_layer", help="pooling_layer", default="rot_pooling", type=str)
    return parser.parse_args()

class TUDataset(InMemoryDataset):

    url = "https://www.chrsmrrs.com/graphkerneldatasets"

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr=False,
        use_edge_attr=False,
        cleaned=False,
    ):
        self.name = name
        self.cleaned = cleaned
        self.num_tasks = 1
        self.task_type = "classification"
        self.eval_metric = "accuracy"
        super(TUDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self):
        name = "raw{}".format("_cleaned" if self.cleaned else "")
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = "processed{}".format("_cleaned" if self.cleaned else "")
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ["A", "graph_indicator"]
        return ["{}_{}.txt".format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url("{}/{}.zip".format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return "{}({})".format(self.name, len(self))


class TUEvaluator:
    def __init__(self):
        self.num_tasks = 1
        self.eval_metric = "accuracy"

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == "accuracy":
            if not "y_true" in input_dict:
                raise RuntimeError("Missing key of y_true")
            if not "y_pred" in input_dict:
                raise RuntimeError("Missing key of y_pred")

            y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

            """
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            """

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()


            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError("Arguments to Evaluator need to be either numpy ndarray or torch tensor")

            if not y_true.shape == y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same")

            if not y_true.ndim == 2:
                raise RuntimeError(
                    "y_true and y_pred mush to 2-dim arrray, {}-dim array given".format( y_true.ndim)
                )

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError(
                    "Number of tasks should be {} but {} given".format(self.num_tasks, y_true.shape[1])
                )

            return y_true, y_pred
        else:
            raise ValueError("Undefined eval metric %s " % self.eval_metric)

    def _eval_accuracy(self, y_true, y_pred):
        """
        compute Accuracy score averaged across tasks
        """
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            acc = accuracy_score(y_true[is_labeled], y_pred[is_labeled])
            acc_list.append(acc)

        return {"accuracy": sum(acc_list) / len(acc_list)}

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        return self._eval_accuracy(y_true, y_pred)


def initialize_edge_weight(data):
    data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
    return data


def initialize_node_features(data):
    num_nodes = int(data.edge_index.max()) + 1
    data.x = torch.ones((num_nodes, 1))
    return data


def set_tu_dataset_y_shape(data):
    num_tasks = 1
    data.y = data.y.unsqueeze(num_tasks)
    return data


class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, mlp_edge_model_dim=64):
        super(ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = self.encoder.out_node_dim

        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1),
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr):

        _, node_emb = self.encoder(batch, x, edge_index, edge_attr)

        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits


def get_emb_y(loader, encoder, device, dtype="numpy", is_rand_label=False):
    x, y = encoder.get_embeddings(loader, device, is_rand_label)
    if dtype == "numpy":
        return x, y
    elif dtype == "torch":
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    else:
        raise NotImplementedError


class EmbeddingEvaluation:
    def __init__(
        self,
        base_classifier,
        evaluator,
        task_type,
        num_tasks,
        device,
        params_dict=None,
        param_search=False,
        is_rand_label=False,
    ):
        self.is_rand_label = is_rand_label
        self.base_classifier = base_classifier
        self.evaluator = evaluator
        self.eval_metric = evaluator.eval_metric
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.device = device
        self.param_search = param_search
        self.params_dict = params_dict
        if self.eval_metric == "rmse":
            self.gscv_scoring_name = "neg_root_mean_squared_error"
        elif self.eval_metric == "mae":
            self.gscv_scoring_name = "neg_mean_absolute_error"
        elif self.eval_metric == "rocauc":
            self.gscv_scoring_name = "roc_auc"
        elif self.eval_metric == "accuracy":
            self.gscv_scoring_name = "accuracy"
        else:
            raise ValueError(
                "Undefined grid search scoring for metric %s " % self.eval_metric
            )

        self.classifier = None

    def scorer(self, y_true, y_raw):
        input_dict = {"y_true": y_true, "y_pred": y_raw}
        score = self.evaluator.eval(input_dict)[self.eval_metric]
        return score

    def ee_binary_classification(
        self, train_emb, train_y, val_emb, val_y, test_emb, test_y
    ):
        # param_search = False
        if self.param_search:
            params_dict = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            self.classifier = make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    self.base_classifier,
                    params_dict,
                    cv=5,
                    scoring=self.gscv_scoring_name,
                    n_jobs=16,
                    verbose=0,
                ),
            )
        else:
            self.classifier = make_pipeline(StandardScaler(), self.base_classifier)

        self.classifier.fit(train_emb, np.squeeze(train_y))

        if self.eval_metric == "accuracy":
            train_raw = self.classifier.predict(train_emb)
            val_raw = self.classifier.predict(val_emb)
            test_raw = self.classifier.predict(test_emb)
        else:
            train_raw = self.classifier.predict_proba(train_emb)[:, 1]
            val_raw = self.classifier.predict_proba(val_emb)[:, 1]
            test_raw = self.classifier.predict_proba(test_emb)[:, 1]

        return (
            np.expand_dims(train_raw, axis=1),
            np.expand_dims(val_raw, axis=1),
            np.expand_dims(test_raw, axis=1),
        )

    def ee_multioutput_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):

        self.classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(self.base_classifier, n_jobs=-1))

        if np.isnan(train_y).any():
            print("Has NaNs ... ignoring them")
            train_y = np.nan_to_num(train_y)
        self.classifier.fit(train_emb, train_y)

        train_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(train_emb)])
        val_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(val_emb)])
        test_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(test_emb)])

        return train_raw, val_raw, test_raw

    def ee_regression(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if self.param_search:
            params_dict = {"alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
            self.classifier = GridSearchCV(
                self.base_classifier,
                params_dict,
                cv=5,
                scoring=self.gscv_scoring_name,
                n_jobs=16,
                verbose=0,
            )
        else:
            self.classifier = self.base_classifier

        self.classifier.fit(train_emb, np.squeeze(train_y))

        train_raw = self.classifier.predict(train_emb)
        val_raw = self.classifier.predict(val_emb)
        test_raw = self.classifier.predict(test_emb)

        return (
            np.expand_dims(train_raw, axis=1),
            np.expand_dims(val_raw, axis=1),
            np.expand_dims(test_raw, axis=1),
        )

    def embedding_evaluation(self, encoder, train_loader, valid_loader, test_loader):
        encoder.eval()
        train_emb, train_y = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label)
        val_emb, val_y = get_emb_y(valid_loader, encoder, self.device, is_rand_label=self.is_rand_label)
        test_emb, test_y = get_emb_y(test_loader, encoder, self.device, is_rand_label=self.is_rand_label)
        if "classification" in self.task_type:
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, train_y, val_emb, val_y, test_emb, test_y)
            elif self.num_tasks > 1:
                (train_raw, val_raw, test_raw,) = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y, test_emb, test_y)
            else:
                raise NotImplementedError
        else:
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
            else:
                raise NotImplementedError

        train_score = self.scorer(train_y, train_raw)

        val_score = self.scorer(val_y, val_raw)

        test_score = self.scorer(test_y, test_raw)

        return train_score, val_score, test_score
    #我改了这里的128 为4
    def kf_embedding_evaluation(self, encoder, dataset, folds=10, batch_size=128):
        kf_train = []
        kf_val = []
        kf_test = []

        kf = KFold(n_splits=folds, shuffle=True, random_state=None)
        for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
            test_dataset = [dataset[int(i)] for i in list(test_index)]
            train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)

            train_dataset = [dataset[int(i)] for i in list(train_index)]
            val_dataset = [dataset[int(i)] for i in list(val_index)]

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            train_score, val_score, test_score = self.embedding_evaluation(encoder, train_loader, valid_loader, test_loader)

            kf_train.append(train_score)
            kf_val.append(val_score)
            kf_test.append(test_score)

        return (
            np.array(kf_train).mean(),
            np.array(kf_val).mean(),
            np.array(kf_test).mean(),
        )


def reset(nn):
    def _reset(item):
        if hasattr(item, "reset_parameters"):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, "children") and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class WGINConv(MessagePassing):
    def __init__(
        self, nn: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(WGINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self,  x: Union[Tensor, OptPairTensor], edge_index: Adj,  edge_weight=None, size: Size = None,) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)



class TUEncoder(torch.nn.Module):
    def __init__(self, num_dataset_features, pooling_layer, f_method, a1, a2, a3, rho, same_para, p0, q0, num, eps, h, emb_dim=300, num_gc_layers=5, drop_ratio=0.0,
        pooling_type="standard",
        is_infograph=False,
    ):
        super(TUEncoder, self).__init__()
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
        self.h = h
        self.rho = rho
        self.pooling_type = pooling_type
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph
        self.pooling_layer = (pooling_layer,)
        self.out_node_dim = self.emb_dim
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            self.out_graph_dim = self.emb_dim * self.num_gc_layers
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(
                    Linear(num_dataset_features, emb_dim),
                    ReLU(),
                    Linear(emb_dim, emb_dim),
                )
            conv = WGINConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)
        self.pooling_layer = pooling_layer
        feature_pooling = emb_dim
        if self.pooling_layer == "mix_pooling":
            self.pooling = Pooling.MixedPooling()
        if self.pooling_layer == "gated_pooling":
            self.pooling = Pooling.GatedPooling(feature_pooling)
        if self.pooling_layer == "set_set":
            self.pooling = Pooling.Set2Set(feature_pooling, 2, 1)
            self.dense = torch.nn.Linear(feature_pooling * 2, 32)
        if self.pooling_layer == "attention_pooling":
            self.pooling = Pooling.AttentionPooling(feature_pooling, 32)
        if self.pooling_layer == "gated_attention_pooling":
            self.pooling = Pooling.GatedAttentionPooling(feature_pooling, 32)
        if self.pooling_layer == "dynamic_pooling":
            self.pooling = Pooling.DynamicPooling(feature_pooling, 3)
        if self.pooling_layer == "uot_pooling":
            self.pooling = Pooling.UOTPooling(dim=feature_pooling, num=num, rho = rho, same_para=same_para, p0=p0, q0=q0, eps=eps, a1=a1, a2=a2, a3=a3, f_method=args.f_method)
        if self.pooling_layer == "rot_pooling":
            self.pooling = Pooling.ROTPooling(dim=feature_pooling, num=num,  rho = rho, same_para=same_para, p0=p0, q0=q0, eps=eps, a1=a1, a2=a2, a3=a3, f_method=args.f_method)
        if self.pooling_layer == "deepset":
            self.pooling = Pooling.DeepSet(feature_pooling, 32)
        if self.pooling_layer == 'GeneralizedNormPooling':
            self.pooling = Pooling.GeneralizedNormPooling(feature_pooling)
        if self.pooling_layer == 'SAGPooling':
            self.pooling = SAGPooling(feature_pooling)
        if self.pooling_layer == 'ASAPooling':
            self.pooling = ASAPooling(feature_pooling)

    def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        # compute graph embedding using pooling
        if self.pooling_type == "standard":
            if self.pooling_layer == "add_pooling":
                xpool = global_add_pool(x, batch)
            elif self.pooling_layer == "mean_pooling":
                xpool = global_mean_pool(x, batch)
                xpool = xpool
            elif self.pooling_layer == "max_pooling":
                xpool = global_max_pool(x, batch)
            elif self.pooling_layer == "set_set":
                torch.backends.cudnn.enabled = False
                xpool = self.pooling(x, batch)
                xpool = self.dense(xpool)
            elif self.pooling_layer == 'SAGPooling':
                xpool, _, _, batch, _, _ = self.pooling(x, edge_index, batch=batch)
                xpool = global_add_pool(xpool, batch)
            elif self.pooling_layer == 'ASAPooling':
                xpool, _, _, batch, _ = self.pooling(x, edge_index, batch=batch)
                xpool = global_add_pool(xpool, batch)
            else:
                xpool = self.pooling(x, batch)
                #logs_temp = np.array(xpool.cpu())
            #print(xpool)
            return xpool, x

        elif self.pooling_type == "layerwise":
            if self.pooling_layer == "add_pooling":
                xpool = [global_add_pool(x, batch) for x in xs]
            elif self.pooling_layer == "mean_pooling":
                xpool = [global_mean_pool(x, batch) for x in xs]
            elif self.pooling_layer == "max_pooling":
                xpool = [global_max_pool(x, batch) for x in xs]
            # for set_set
            elif self.pooling_layer == "set_set":
                torch.backends.cudnn.enabled = False
                xpool = [self.pooling(x, batch) for x in xs]
                xpool = [self.dense(x2) for x2 in xpool]
            else:
                xpool = [self.pooling(x, batch) for x in xs]
            xpool = torch.cat(xpool, 1)
            if self.is_infograph:
                return xpool, torch.cat(xs, 1)
            else:
                return xpool, x
        else:
            raise NotImplementedError

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(batch, x, edge_index, edge_weight)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class GInfoMinMax(torch.nn.Module):
    def __init__(self, encoder, proj_hidden_dim=300):
        super(GInfoMinMax, self).__init__()

        self.encoder = encoder
        self.input_proj_dim = self.encoder.out_graph_dim

        self.proj_head = Sequential(
            Linear(self.input_proj_dim, proj_hidden_dim),
            ReLU(inplace=True),
            Linear(proj_hidden_dim, proj_hidden_dim),
        )

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):

        z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)
        z = self.proj_head(z)
        return z, node_emb

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x, x_aug) / torch.einsum("i,j->ij", x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = -torch.log(loss_0).mean()
            loss_1 = -torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = -torch.log(loss_1).mean()
            return loss_1

        return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.use_deterministic_algorithms(True)


def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    evaluator = TUEvaluator()
    my_transforms = Compose([initialize_node_features, initialize_edge_weight, set_tu_dataset_y_shape])
    dataset = TUDataset("./original_datasets/", args.DS, transform=my_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    model = GInfoMinMax(
        TUEncoder(
            num_dataset_features=1,
            pooling_layer=args.pooling_layer,
            a1=args.a1,
            a2=args.a2,
            a3=args.a3,
            f_method=args.f_method,
            rho=args.rho,
            same_para=args.same_para,
            p0=args.p0,
            q0=args.q0,
            num=args.num,
            eps=args.eps,
            h=args.h,
            emb_dim=args.emb_dim,
            num_gc_layers=args.num_gc_layers,
            drop_ratio=args.drop_ratio,
            pooling_type=args.pooling_type,
        ),
        args.emb_dim,
    ).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
    view_learner = ViewLearner(
        TUEncoder(
            num_dataset_features=1,
            pooling_layer=args.pooling_layer,
            a1=args.a1,
            a2=args.a2,
            a3=args.a3,
            f_method=args.f_method,
            rho=args.rho,
            same_para=args.same_para,
            p0=args.p0,
            q0=args.q0,
            num=args.num,
            eps=args.eps,
            h=args.h,
            emb_dim=args.emb_dim,
            num_gc_layers=args.num_gc_layers,
            drop_ratio=args.drop_ratio,
            pooling_type=args.pooling_type,
        ),
        mlp_edge_model_dim=args.mlp_edge_model_dim,
    ).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)
    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(
            LinearSVC(dual=False, fit_intercept=True),
            evaluator,
            dataset.task_type,
            dataset.num_tasks,
            device,
            param_search=False,
        )
    else:
        ee = EmbeddingEvaluation(
            SVC(),
            evaluator,
            dataset.task_type,
            dataset.num_tasks,
            device,
            param_search=False,
        )
    model.eval()
    train_score, val_score, test_score = ee.kf_embedding_evaluation(
        model.encoder, dataset
    )
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score, test_score)
    )

    model_losses = []
    view_losses = []
    view_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []

    accuracies = {"val": [], "test": []}
    for epoch in range(1, args.epochs + 1):
        begin_time = time.time()
        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0
        for batch in dataloader:
            batch = batch.to(device)

            view_learner.train()
            view_learner.zero_grad()
            model.eval()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)

            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            x_aug, _ = model(
                batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight
            )

            # regularization
            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            reg = torch.stack(reg)
            reg = reg.mean()

            view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg)
            view_loss_all += view_loss.item() * batch.num_graphs
            reg_all += reg.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()

            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug, _ = model(
                batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight
            )

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()

        end_time = time.time()
        print("-------------time------------")
        print(end_time - begin_time)
        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)

        logging.info(
            "Epoch {}, Model Loss {}, View Loss {}, Reg {}".format(epoch, fin_model_loss, fin_view_loss, fin_reg)
        )
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)
        if epoch % args.eval_interval == 0:
            model.eval()

            train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)

            logging.info(
                "Metric: {} Train: {} Val: {} Test: {}".format(evaluator.eval_metric, train_score, val_score, test_score)
            )

            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)
            accuracies["val"].append(val_score)
            accuracies["test"].append(test_score)
    if "classification" in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    logging.info("FinishedTraining!")
    logging.info("BestEpoch: {}".format(best_val_epoch))
    logging.info("BestTrainScore: {}".format(best_train))
    logging.info("BestValidationScore: {}".format(valid_curve[best_val_epoch]))
    logging.info("FinalTestScore: {}".format(test_curve[best_val_epoch]))
    return valid_curve[best_val_epoch], test_curve[best_val_epoch]


if __name__ == "__main__":

    args = arg_parse()
    run(args)

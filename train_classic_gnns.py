"""
Clean training script for GCN, GCNII, APPNP, GAT on node classification.
Fixes:
  - uint8 mask deprecation → all masks cast to bool
  - Removes the erroneous [:,rb] 2D mask indexing for standard datasets
  - Self-contained model definitions (no external models.py dependency)
"""

import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns

from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures

# ─────────────────────────────────────────────
# 1.  MODEL DEFINITIONS
# ─────────────────────────────────────────────

from torch_geometric.nn import GCNConv, GATConv, APPNP as APPNPProp


# ── GCN ──────────────────────────────────────
class GCN_Net(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ── GAT ──────────────────────────────────────
class GAT_Net(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, args.hidden,
                             heads=args.heads, dropout=args.dropout)
        self.conv2 = GATConv(args.hidden * args.heads, dataset.num_classes,
                             heads=args.output_heads, concat=False,
                             dropout=args.dropout)
        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ── APPNP ─────────────────────────────────────
class APPNP_Net(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.prop = APPNPProp(args.K, args.alpha)
        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


# ── GCNII ─────────────────────────────────────
class GCNIILayer(nn.Module):
    """Single GCNII layer: combines residual + identity mapping.
    adj is a dense normalised adjacency matrix (N x N) on the same device.
    """
    def __init__(self, hidden, alpha, beta):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, x0, adj):
        # adj: dense float tensor (N x N), already D^-1/2 (A+I) D^-1/2
        ax = torch.mm(adj, x)                          # graph smoothing
        ax = (1 - self.alpha) * ax + self.alpha * x0  # initial residual
        ax = (1 - self.beta)  * ax + self.beta * self.W(ax)  # identity + W
        return ax


class GCNII(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.lin_in  = nn.Linear(dataset.num_features, args.hidden)
        self.layers  = nn.ModuleList()
        for i in range(1, args.layer_num + 1):
            beta = np.log(args.lamda / i + 1)
            self.layers.append(GCNIILayer(args.hidden, args.alpha, beta))
        self.lin_out = nn.Linear(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

        # Separate param groups for weight-decay scheduling
        self.reg_params     = list(self.layers.parameters())
        self.non_reg_params = list(self.lin_in.parameters()) + list(self.lin_out.parameters())

    def forward(self, data):
        x, adj = data.x, data.adj_dense
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        x0 = x
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x, x0, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)
        return F.log_softmax(x, dim=1)


# ─────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────

PLANETOID = ['Cora', 'Citeseer', 'Pubmed']
WEBKB     = ['Wisconsin', 'Texas', 'Cornell']
WIKI      = ['Chameleon', 'Squirrel']
AMAZON    = ['Computers', 'Photo']
COAUTHOR  = ['CS']


def load_dataset(name, root='data/'):
    if name in PLANETOID:
        return Planetoid(root=root + name, name=name,
                         transform=NormalizeFeatures())
    elif name in WEBKB:
        return WebKB(root=root + name, name=name,
                     transform=NormalizeFeatures())
    elif name in WIKI:
        return WikipediaNetwork(root=root + name, name=name.lower(),
                                transform=NormalizeFeatures())
    elif name == 'Actor':
        return Actor(root=root + name, transform=NormalizeFeatures())
    elif name in AMAZON:
        return Amazon(root=root + name, name=name,
                      transform=NormalizeFeatures())
    elif name in COAUTHOR:
        # Coauthor-CS has no official train/val/test split — random_splits()
        # is used downstream for this dataset, same as WebKB/Wiki/Actor.
        return Coauthor(root=root + name, name=name,
                        transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unknown dataset: {name}")


def random_splits(data, num_classes, percls_trn, val_lb, seed):
    """Create random train/val/test masks (1-D bool tensors)."""
    torch.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        idx = (data.y == i).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]
        indices.append(idx)

    train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_idx  = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_idx  = rest_idx[torch.randperm(rest_idx.size(0))]

    val_idx  = rest_idx[:val_lb]
    test_idx = rest_idx[val_lb:]

    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask
    return data


# ─────────────────────────────────────────────
# 3.  TRAINING / EVALUATION
# ─────────────────────────────────────────────

def train_epoch(model, optimizer, data, dropout):
    model.train()
    optimizer.zero_grad()
    # train_mask is guaranteed bool here
    out  = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data)
    results = {}
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = getattr(data, split).bool()   # ensure bool
        pred = logits[mask].argmax(dim=-1)
        acc  = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        results[split] = (acc, loss)
    return results


def run_experiment(args, dataset, data, Net, run_seed):
    device = args.device

    # ── Rebuild masks for non-Planetoid datasets (or always for clean splits)
    need_random_split = args.dataset not in PLANETOID
    if need_random_split:
        percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
        val_lb     = int(round(args.val_rate * len(data.y)))
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb, run_seed)
    else:
        # Planetoid masks may be 2-D (WebKB style) — collapse to 1-D bool
        for attr in ['train_mask', 'val_mask', 'test_mask']:
            m = getattr(data, attr)
            if m.dim() == 2:
                # use first column as default split
                setattr(data, attr, m[:, 0].bool())
            else:
                setattr(data, attr, m.bool())

    model = Net(dataset, args).to(device)
    data  = data.to(device)

    # ── Optimiser
    if args.net == 'GCNII':
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params,     weight_decay=args.weight_decay1),
            dict(params=model.non_reg_params, weight_decay=args.weight_decay2),
        ], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc  = 0.0
    best_test_acc = 0.0
    val_acc_hist  = []
    time_per_epoch = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_epoch(model, optimizer, data, args.dropout)
        time_per_epoch.append(time.time() - t0)

        res = evaluate(model, data)
        train_acc = res['train_mask'][0]
        val_acc   = res['val_mask'][0]
        test_acc  = res['test_mask'][0]

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc

        val_acc_hist.append(val_acc)

        # Early stopping
        if args.early_stopping > 0 and epoch > args.early_stopping:
            window = torch.tensor(val_acc_hist[-(args.early_stopping + 1):-1])
            if val_acc < window.mean().item():
                print(f'  Early stop at epoch {epoch}')
                break

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f'  Epoch {epoch:4d}  '
                  f'train={train_acc:.4f}  val={val_acc:.4f}  test={test_acc:.4f}')

    return best_test_acc, best_val_acc, time_per_epoch


# ─────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────

NET_MAP = {
    'GCN':   GCN_Net,
    'GAT':   GAT_Net,
    'APPNP': APPNP_Net,
    'GCNII': GCNII,
}

SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660,
         2108550661, 1648766618,  629014539, 3212139042, 2424918363]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',          type=int,   default=123456789)
    parser.add_argument('--epochs',        type=int,   default=1000)
    parser.add_argument('--lr',            type=float, default=0.01)
    parser.add_argument('--weight_decay',  type=float, default=5e-4)
    parser.add_argument('--weight_decay1', type=float, default=1e-2,
                        help='GCNII reg param weight decay')
    parser.add_argument('--weight_decay2', type=float, default=5e-4,
                        help='GCNII non-reg param weight decay')
    parser.add_argument('--early_stopping',type=int,   default=200)
    parser.add_argument('--hidden',        type=int,   default=64)
    parser.add_argument('--dropout',       type=float, default=0.6)

    # APPNP
    parser.add_argument('--K',     type=int,   default=10)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Teleport prob for APPNP / residual for GCNII')

    # GCNII
    parser.add_argument('--lamda',     type=float, default=0.5)
    parser.add_argument('--layer_num', type=int,   default=4,
                        help='Number of GCNII layers')

    # GAT
    parser.add_argument('--heads',        type=int, default=8)
    parser.add_argument('--output_heads', type=int, default=1)

    # Dataset / split
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora','CS', 'Citeseer','Pubmed',
                                 'Chameleon','Squirrel','Actor',
                                 'Wisconsin','Texas','Cornell',
                                 'Computers','Photo'])
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--val_rate',   type=float, default=0.2)

    # Model
    parser.add_argument('--net',  type=str, default='GCN',
                        choices=list(NET_MAP.keys()))
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "auto" | "cpu" | "mps" | "cuda:0"')

    args = parser.parse_args()

    # ── Device  (--device cpu overrides everything)
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            args.device = torch.device('mps')
        else:
            args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)
    print(f'Using device: {args.device}')

    # ── Load data
    dataset = load_dataset(args.dataset)
    data    = dataset[0]

    # ── Pre-compute normalised adjacency as DENSE tensor (for GCNII)
    # Kept on CPU always; moved to device inside run_experiment via data.to(device)
    if args.net == 'GCNII':
        N = data.num_nodes
        edge_index = data.edge_index
        # Add self-loops
        self_loops = torch.arange(N, dtype=torch.long)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        ei = torch.cat([edge_index, self_loops], dim=1)
        # Build dense adjacency
        adj = torch.zeros(N, N)
        adj[ei[0], ei[1]] = 1.0
        # Symmetric normalisation D^-1/2 A D^-1/2
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        data.adj_dense = adj
    else:
        data.adj_dense = None

    # Keep adj_t for legacy compatibility (not used by any model here)
    data.adj_t = None

    Net = NET_MAP[args.net]
    print(args)
    print('─' * 60)

    results     = []
    time_results = []

    for rp in tqdm(range(args.runs), desc='Runs'):
        seed = SEEDS[rp]
        args.seed = seed
        print(f'\nRun {rp + 1}/{args.runs}  seed={seed}')

        test_acc, val_acc, t_per_epoch = run_experiment(
            args, dataset, data.clone(), Net, seed
        )
        results.append([test_acc, val_acc])
        time_results.append(t_per_epoch)
        print(f'  → test_acc={test_acc:.4f}  best_val_acc={val_acc:.4f}')

    # ── Summary
    total_time  = sum(sum(r) for r in time_results)
    total_epochs= sum(len(r) for r in time_results)
    print('\n' + '─' * 60)
    print(f'Avg time per run:   {total_time / args.runs:.2f} s')
    print(f'Avg time per epoch: {1000 * total_time / total_epochs:.2f} ms')

    arr = np.array(results)
    test_mean  = arr[:, 0].mean() * 100
    test_std   = arr[:, 0].std()  * 100
    val_mean   = arr[:, 1].mean() * 100

    values      = arr[:, 0]
    uncertainty = np.max(np.abs(
        sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95)
        - values.mean()
    )) * 100

    print(f'\n{args.net} on {args.dataset} ({args.runs} runs):')
    print(f'  Test  = {test_mean:.2f} ± {uncertainty:.2f}')
    print(f'  Val   = {val_mean:.2f}')
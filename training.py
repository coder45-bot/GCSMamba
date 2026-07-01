import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits
from utils_ordering import random_node_permutation
from torch_sparse import SparseTensor
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
from torch_geometric.datasets import Planetoid
import time
from torch_geometric.utils import to_dense_adj, add_self_loops, remove_self_loops


def generate_geom_splits(data, num_classes, num_runs=10, train_per=0.6, val_per=0.2):
    """
    Generates standard Geom-GCN / H2GCN random splits (60% Train / 20% Val / 20% Test)
    structured as 2D tensors [num_nodes, num_runs] to safely match repeated run allocations.
    """
    num_nodes = data.x.shape[0]
    
    train_masks = torch.zeros((num_nodes, num_runs), dtype=torch.bool)
    val_masks = torch.zeros((num_nodes, num_runs), dtype=torch.bool)
    test_masks = torch.zeros((num_nodes, num_runs), dtype=torch.bool)
    
    for run in range(num_runs):
        for c in range(num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            if idx.size(0) == 0:
                continue
            idx = idx[torch.randperm(idx.size(0))]
            
            num_train = int(idx.size(0) * train_per)
            num_val = int(idx.size(0) * val_per)
            
            train_masks[idx[:num_train], run] = True
            val_masks[idx[num_train:num_train+num_val], run] = True
            test_masks[idx[num_train+num_val:], run] = True
            
    return train_masks, val_masks, test_masks


def RunExp(args, dataset, data, Net, rb, val_lb):

    def train(args, model, optimizer, data, dprate, rb):
        model.train()
        optimizer.zero_grad()

        # Handle both 1D and 2D train masks
        if data.train_mask.dim() == 2:
            train_mask = data.train_mask[:, rb]
        else:
            train_mask = data.train_mask

        out_main = model(data)[train_mask]
        nll_main = F.nll_loss(out_main, data.y[train_mask])

        loss = nll_main
        loss.backward()
        optimizer.step()

        del out_main

    def test(args, model, data, rb):
        model.eval()

        logits_main = model(data)

        accs = []
        losses = []
        preds = []

        for _, mask in data('train_mask', 'val_mask', 'test_mask'):

            # Handle both 1D and 2D masks
            if mask.dim() == 2:
                cur_mask = mask[:, rb]
            else:
                cur_mask = mask

            pred = logits_main[cur_mask].max(dim=-1)[1]

            acc = (
                pred.eq(data.y[cur_mask])
                .sum()
                .item()
                / cur_mask.sum().item()
            )

            loss = F.nll_loss(
                logits_main[cur_mask],
                data.y[cur_mask]
            )

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())

        return accs, preds, losses
    
    def normalize_adj_tensor(adj):
        mx = adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    tmp_net = Net(dataset, args)

    # Note: If splits are already generated, we avoid overwriting them here for non-Planetoid datasets
    if args.dataset not in ['Cora', 'Citeseer', 'Pubmed', 'Chameleon', 'Squirrel', 'Actor', 'Wisconsin', 'Texas', 'Cornell', 'CS', 'Physics']:
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)

    model, data = tmp_net.to(device), data.to(device)

    if args.net == 'GPRGNN':
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}
        ])
    elif args.net == 'GCN_mamba_Net_New':
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.mamba.in_proj.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.mamba.A_log, 'weight_decay': 0.01, 'lr': args.lr},
            {'params': model.mamba.D, 'weight_decay': 0.01, 'lr': args.lr},
            {'params': model.mamba.x_proj.parameters(), 'weight_decay': 0.01, 'lr': args.lr},
            {'params': model.mamba.dt_proj.parameters(), 'weight_decay': 0.01, 'lr': args.lr},
            {'params': model.mamba.out_proj.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.norm_1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.norm_2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.mamba.in_act_net.parameters(), 'weight_decay': args.weight_decay, 'lr': args.Stat_lr},
            {'params': model.mamba.out_act_net.parameters(), 'weight_decay': args.weight_decay, 'lr': args.Stat_lr}
        ])
    elif args.net == 'GCNII':
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=args.weight_decay1),
            dict(params=model.non_reg_params, weight_decay=args.weight_decay2)
        ], lr=args.lr)
    elif args.net == 'HybridTopoMamba_Net':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        # FAGCN_Net, H2GCN_Net, ACMGCN_Net and GMN_Net optimize using standard parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        train(args, model, optimizer, data, args.dprate, rb)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(args, model, data, rb)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(val_acc_history[-(args.early_stopping + 1):-1])
                if val_acc < tmp.mean().item():
                    print('The sum of epochs:', epoch)
                    break
        print('train_acc:{:.4f},val_acc:{:.4f},temp_test_acc:{:.4f}'.format(train_acc, val_acc, tmp_test_acc))
    return test_acc, best_val_acc, theta, time_run, model


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# adj normalization
def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123456789, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=5000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')     
    parser.add_argument('--Stat_lr', type=float, default=0.05, help='State learning rate.')   
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout for neural networks.')
    
    # parameters for GCNII
    parser.add_argument('--lamda', type=float, default=0.5, help='propagation steps.')
    parser.add_argument('--weight_decay1', type=float, default=5e-4, help='weight decay.')
    parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--dprate', type=float, default=0.0, help='dropout for propagation layer.')
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR', help='initialization for GPRGNN.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    parser.add_argument('--dataset', type=str, choices=['Wisconsin', 'Chameleon', 'Texas', 'Cornell', 'Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Actor', 'CS', 'Physics'],
                        default='Cora')
    parser.add_argument('--device', type=int, default=2, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    
    # Added FAGCN_Net, H2GCN_Net, and ACMGCN_Net to valid argument options
    parser.add_argument('--net', type=str, choices=['SSGC_Net', 'SGC_Net', 'GCNII', 'GCN_mamba_Net', 'HybridTopoMamba_Net', 'GMN_Net', 'GCN', 'GAT', 'APPNP', 'GPRGNN', 'MLP', 'FAGCN_Net', 'H2GCN_Net', 'ACMGCN_Net'], default='GCN_mamba_Net')

    # parameters for MbaGCN
    parser.add_argument('--d_model', type=int, default=64, help='hidden units.')
    parser.add_argument('--d_inner', type=int, default=64, help='')
    parser.add_argument('--dt_rank', type=int, default=4, help='')
    parser.add_argument('--d_state', type=int, default=4, help='')
    parser.add_argument('--ctx_dim', type=int, default=8, help='Graph context dimension')
    parser.add_argument('--freeze_gate_off', action='store_true', help='Disable topology conditioning')
    parser.add_argument('--bias', type=bool, default=True, help='')
    parser.add_argument('--mamba_dropout', type=float, default=0.8, help='')
    parser.add_argument('--layer_num', type=int, default=3, help='')
    parser.add_argument('--d_conv', type=int, default=4, help='')
    parser.add_argument('--use_gce', action='store_true', default=True)
    parser.add_argument('--no_use_gce', dest='use_gce', action='store_false')
    parser.add_argument('--ordering', type=str, default='default', choices=['default', 'random'], help='Node ordering used before the Mamba scan.')

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args.device = device

    # 10 fixed seeds for splits
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042, 2424918363]

    print(args)
    print("---------------------------------------------")

    dataset = DataLoader(args.dataset)
    data = dataset[0]
    
    # --- FIXED SECTION FOR DATASETS MISSING DEFAULT MASKS (like CS / Physics) ---
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else int(data.y.max() + 1)
        print(f"No default masks found for dataset: {args.dataset}. Generating 60/20/20 splits...")
        
        train_mask, val_mask, test_mask = generate_geom_splits(
            data=data, 
            num_classes=num_classes, 
            num_runs=args.runs, 
            train_per=args.train_rate, 
            val_per=args.val_rate
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    print("Dataset:", args.dataset)
    print("train_mask:", data.train_mask.shape)
    print("val_mask:", data.val_mask.shape)
    print("test_mask:", data.test_mask.shape)
    
    if args.ordering == "random":
        print("Using RANDOM node ordering")
        data = random_node_permutation(data)
    else:
        print("Using DEFAULT node ordering")
    

    adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes))
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    if args.net == 'SGC_Net':
        x = data.x
        for i in range(10):
            x = torch.mm(adj_t.to_dense(), x)
        data.x = x
    elif args.net == 'SSGC_Net':
        alpha = 0.05
        x = data.x
        emb = alpha * x
        degree = 10
        for i in range(degree):
            x = torch.mm(adj_t.to_dense(), x)
            emb = emb + (1 - alpha) * x / degree
        data.x = emb

    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))

    results = []
    thetas = []
    time_results = []
    
    for RP in tqdm(range(args.runs)):

        args.seed = SEEDS[RP]
        gnn_name = args.net
        if gnn_name == 'GCN':
            Net = GCN_Net
        elif gnn_name == 'GAT':
            Net = GAT_Net
        elif gnn_name == 'APPNP':
            Net = APPNP_Net
        elif gnn_name == 'GPRGNN':
            Net = GPRGNN
        elif gnn_name == 'MLP':
            Net = MLP
        elif gnn_name == 'GCNII':
            Net = GCNII
        elif gnn_name == 'HybridTopoMamba_Net':
            Net = HybridTopoMamba_Net
        elif gnn_name == 'SGC_Net':
            Net = SGC_Net
        elif gnn_name == 'SSGC_Net':
            Net = SGC_Net
        elif gnn_name == 'GCN_mamba_Net':
            Net = GCN_mamba_Net
        elif gnn_name == 'GMN_Net':
            Net = GMN_Net
        elif gnn_name == 'FAGCN_Net':
            Net = FAGCN_Net
        elif gnn_name == 'H2GCN_Net':
            Net = H2GCN_Net
        elif gnn_name == 'ACMGCN_Net':
            Net = ACMGCN_Net

        test_acc, best_val_acc, theta_0, time_run, tmp_net = RunExp(args, dataset, data, Net, RP, val_lb)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        thetas.append(theta_0)
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
        if args.net == 'BernNet':
            print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / (args.runs), "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values = np.asarray(results)[:, 0]
    uncertainty = np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.2f}±{uncertainty*100:.2f}   \t val acc mean = {val_acc_mean:.2f}')
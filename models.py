import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from GCNIIlayer import GCNIIdenseConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Module, Linear, Parameter, ModuleList, Dropout
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch.nn import Linear
from torch_geometric.nn import FAConv
from torch import Tensor
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv, ChebConv, GCN2Conv, SGConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.special import comb
from torch_sparse import SparseTensor
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch.nn import Linear, ModuleList, Module, Dropout, ReLU, GELU, Sequential
from einops import rearrange, repeat, einsum


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GCNII(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNII, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(dataset.num_features, args.hidden))
        for _ in range(args.layer_num):
            self.convs.append(GCNIIdenseConv(args.hidden, args.hidden))
        self.dropout = args.dropout
        self.lamda = args.lamda
        self.alpha = args.alpha
        self.convs.append(torch.nn.Linear(args.hidden, dataset.num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, self.dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout ,training=self.training)
            beta = math.log(self.lamda/(i+1)+1)
            x = F.relu(con(x, edge_index, self.alpha, _hidden[0], beta, edge_weight))
        x = F.dropout(x, self.dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)

class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        
class ActionNet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GCN_mamba_liner(args.d_inner, 2, with_bias=True)
        self.dropout = torch.nn.Dropout(0.6)
        self.act = torch.nn.ReLU()
        self.reset_parameters()
    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, x) -> Tensor:

        x = self.dropout(self.net(x))

        return x
        
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SGC_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SGC_Net, self).__init__()
        self.lin1 = MLP(dataset, args)
    
    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, data):
        x = self.lin1(data)
        return x

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.conv = torch.nn.ModuleList()
        self.dataset = args.dataset
        self.hidden_layer_num = 1
        for i in range(self.hidden_layer_num):
            self.conv.append(GCNConv(args.hidden, args.hidden))
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        if self.dataset in ['arxiv']:
            x, edge_index = data.x, data.adj_t
        else:
            x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.hidden_layer_num):
            x = F.relu(self.conv[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class MolConv(MessagePassing):
    def __init__(self, aggr='add'):
        super().__init__(aggr=aggr)

    def message(self, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor = None) -> Tensor:
        if edge_attr is None:
            if edge_weight is None:
                return x_j
            else:
                return edge_weight.view(-1, 1) * x_j
        else:
            if edge_weight is None:
                return x_j + edge_attr
            else:
                return edge_weight.view(-1, 1) * (x_j + edge_attr)

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out

class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout =args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    
class GCN_mamba_liner(torch.nn.Module):
    def __init__(self, in_features, out_features, with_bias=False):
        super(GCN_mamba_liner, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class GCN_mamba_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_mamba_Net, self).__init__()
        self.dropout = args.mamba_dropout
        self.args = args
        self.lin1 = GCN_mamba_liner(dataset.num_features, args.d_model, with_bias=args.bias)
        self.layer_num = args.layer_num
        self.mamba = GCN_mamba_block(args)
        self.norm_1 = RMSNorm(args.d_model)
        self.norm_2 = RMSNorm(args.d_model)
        self.lin2 = GCN_mamba_liner(args.d_model, dataset.num_classes, with_bias=args.bias)
        self.reset_parameters()
    def getmamba(self):
        return self.mamba
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x_input = data.x 
        if self.args.dataset in ['arxiv']:
            edge_index = data.adj_t
        else:
            edge_index = data.edge_index

        x = F.relu(self.lin1(x_input))
        x = F.dropout(x, p=self.dropout, training=self.training)
        output = self.mamba(self.norm_1(x), edge_index, data.adj_t, self.args.layer_num) + x
        y = self.norm_2(output)
        y = self.lin2(y)
        return F.log_softmax(y, dim=-1)
    
    def normalize_adj_tensor(self, adj):
        mx = adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
    
class GCN_mamba_block(torch.nn.Module):
    def __init__(self, args):
        super(GCN_mamba_block, self).__init__()
        self.args = args
        self.in_proj = GCN_mamba_liner(args.d_model, args.d_inner * 2, with_bias=args.bias)
        self.x_proj = GCN_mamba_liner(args.d_inner, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.dropout = args.dropout
        self.dt_proj = GCN_mamba_liner(args.dt_rank, args.d_inner, with_bias=args.bias)
        self.bns = torch.nn.ModuleList()

        for i in range(args.layer_num-1):
            self.bns.append(torch.nn.BatchNorm1d(args.d_inner))

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = GCN_mamba_liner(args.d_inner, args.d_model, with_bias=args.bias)
        self.in_act_net = ActionNet(args)
        self.out_act_net = ActionNet(args)
        self.reset_parameters()

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.in_act_net.reset_parameters()
        self.out_act_net.reset_parameters()

    def forward(self, x, edge_index, adj, layer_num):
        (b, d) = x.shape
        expanded_x = x.unsqueeze(1)
        x = expanded_x.expand(b, layer_num, d)
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = F.relu(x)
        y = self.ssm(x, self.args, edge_index, adj)
        y = y * F.relu(res)
        output = self.out_proj(y)
        return output[:,-1,:]
    
    def ssm(self, x, args, edge_index, adj):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, edge_index, delta, A, B, C, D, args, adj)
        return y

    def selective_scan(self, u, edge_index, delta, A, B, C, D, args, adj):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = [] 
        for i in range(args.layer_num):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x.reshape(b, d_in * n)
            x = adj @ x
            x = x.reshape(b, d_in, n)
            in_logits = self.in_act_net(x=y)
            out_logits = self.out_act_net(x=y)
            temp = 0.01
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(len(y), len(y)))
            adj = adj.set_diag()
            deg = adj.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y
    
    def create_edge_weight(self, edge_index, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class GraphContextEncoder(Module):
    def __init__(self, in_ch: int, ctx_dim: int):
        super(GraphContextEncoder, self).__init__()
        self.proj = Linear(in_ch * 2 + 1, ctx_dim)

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
        deg_feat = torch.log1p(deg).unsqueeze(-1)
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])
        deg_safe = deg.clamp(min=1.0).unsqueeze(-1)
        neighbor_mean = neighbor_sum / deg_safe
        neighbor_max = torch.full_like(x, float("-inf"))
        neighbor_max.scatter_reduce_(
            0, col.unsqueeze(1).expand(-1, x.size(1)), x[row], reduce="amax", include_self=True
        )
        neighbor_max = torch.where(torch.isinf(neighbor_max), torch.zeros_like(neighbor_max), neighbor_max)
        ctx_input = torch.cat([deg_feat, neighbor_mean, neighbor_max], dim=-1)
        return self.proj(ctx_input)


def build_norm_adj(edge_index: Tensor, num_nodes: int, device) -> SparseTensor:
    row, col = edge_index
    self_loop_idx = torch.arange(num_nodes, device=device)
    full_row = torch.cat([row, col, self_loop_idx])
    full_col = torch.cat([col, row, self_loop_idx])
    full_val = torch.ones(full_row.size(0), device=device)
    adj = SparseTensor(
        row=full_row, col=full_col, value=full_val,
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)


class TopoConditionedPropagationScan(Module):
    def __init__(self, d_model: int, d_state: int, ctx_dim: int, num_steps: int,
                 dropout: float = 0.0, freeze_gate_off: bool = False):
        super(TopoConditionedPropagationScan, self).__init__()
        self.d_model, self.d_state, self.num_steps = d_model, d_state, num_steps
        self.in_proj  = Linear(d_model, d_model * 2)
        self.x_proj   = Linear(d_model, d_state * 2)
        self.dt_proj  = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.A_log = Parameter(torch.log(A))
        self.D = Parameter(torch.ones(d_model))
        self.W_delta_g = Linear(ctx_dim, d_model)
        self.W_A_g = Linear(ctx_dim, d_model * d_state)
        self.freeze_gate_off = freeze_gate_off
        if freeze_gate_off:
            self.register_buffer("ctx_gate", torch.tensor(-15.0))
        else:
            self.ctx_gate = Parameter(torch.tensor(0.0))
        self.dropout = Dropout(dropout)

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.W_delta_g.reset_parameters()
        self.W_A_g.reset_parameters()

    def forward(self, x: Tensor, g: Tensor, norm_adj: SparseTensor) -> Tensor:
        N, d_model = x.shape
        gate = torch.sigmoid(self.ctx_gate)
        x_seq = x.unsqueeze(1).expand(N, self.num_steps, d_model).contiguous()
        x_and_res = self.in_proj(x_seq)
        x_in, res = x_and_res.split(d_model, dim=-1)
        x_in = F.relu(x_in)
        x_dbl = self.x_proj(x_in)
        B, C = x_dbl.split(self.d_state, dim=-1)
        delta_base = F.softplus(self.dt_proj(x_in))
        delta_g = self.W_delta_g(g)
        A_g = self.W_A_g(g).view(N, d_model, self.d_state)
        A_base = -torch.exp(self.A_log)
        state = torch.zeros(N, d_model, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(self.num_steps):
            u_t = x_in[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]
            delta_t = delta_base[:, t, :] + gate * delta_g
            A_t = A_base.unsqueeze(0) - gate * F.softplus(A_g)
            dA = torch.exp(delta_t.unsqueeze(-1) * A_t)
            dBu = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)
            state = dA * state + dBu
            y_t = torch.einsum('ndk,nk->nd', state, C_t)
            state = self.dropout(state)
            state_flat = state.reshape(N, d_model * self.d_state)
            state_flat = norm_adj @ state_flat
            state = state_flat.reshape(N, d_model, self.d_state)
            ys.append(y_t.unsqueeze(1))
        y = torch.cat(ys, dim=1)
        y = y + x_in * self.D
        y = y * F.relu(res)
        out = self.out_proj(y)
        return out[:, -1, :]


class HybridTopoMamba_Net(Module):
    def __init__(self, dataset, args):
        super(HybridTopoMamba_Net, self).__init__()
        self.dataset = args.dataset
        self.dropout = args.dropout
        ctx_dim = getattr(args, "ctx_dim", 8)
        d_state = getattr(args, "d_state", 16)
        self.input_proj = Linear(dataset.num_features, args.hidden)
        self.ctx_encoder = GraphContextEncoder(args.hidden, ctx_dim)
        self.norm_in = RMSNorm(args.hidden)
        self.scan = TopoConditionedPropagationScan(
            d_model=args.hidden, d_state=d_state, ctx_dim=ctx_dim,
            num_steps=args.layer_num, dropout=args.dropout,
            freeze_gate_off=getattr(args, "freeze_gate_off", False),
        )
        self.norm_out = RMSNorm(args.hidden)
        self.output_proj = Linear(args.hidden, dataset.num_classes)
        self._cached_adj = None
        self._cached_key = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.ctx_encoder.reset_parameters()
        self.scan.reset_parameters()
        self.output_proj.reset_parameters()

    def _get_norm_adj(self, edge_index: Tensor, num_nodes: int) -> SparseTensor:
        key = (edge_index.data_ptr(), num_nodes)
        if self._cached_key != key:
            self._cached_adj = build_norm_adj(edge_index, num_nodes, edge_index.device)
            self._cached_key = key
        return self._cached_adj

    def forward(self, data):
        x = data.x
        if self.dataset in ['arxiv']:
            edge_index = data.adj_t
        else:
            edge_index = data.edge_index
        num_nodes = x.size(0)
        norm_adj = self._get_norm_adj(edge_index, num_nodes)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.input_proj(h))
        g = self.ctx_encoder(h, edge_index, num_nodes)
        h_normed = self.norm_in(h)
        scan_out = self.scan(h_normed, g, norm_adj)
        h = h + scan_out
        h = self.norm_out(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.output_proj(h)
        return F.log_softmax(out, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# FIXED GMN IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────────────

class GMN_Block(torch.nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        # SSM representation updater representation layer
        self.mamba = torch.nn.Linear(hidden_dim, hidden_dim) 
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, adj):
        # 1. Graph Diffusion Step (Using the pre-normalized SparseTensor natively)
        x_diff = adj @ x
        
        # 2. SSM State & Nonlinear Representation Update with Residual connection
        residual = x_diff
        x = self.norm(x_diff)
        x = self.mamba(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual

class GMN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GMN_Net, self).__init__()
        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        hidden_channels = args.d_model
        num_layers = args.layer_num
        
        # FIX: Using standard node dropout (args.dropout) to avoid the extreme 0.8 loss of input features
        self.dropout = args.dropout
        
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList([GMN_Block(hidden_channels, dropout=self.dropout) for _ in range(num_layers)])
        self.output_proj = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.layers:
            layer.mamba.reset_parameters()
        self.output_proj.reset_parameters()

    def forward(self, data):
        x = data.x
        # FIX: Use precomputed symmetrically normalized sparse adjacency with self-loops
        adj = data.adj_t 
        
        # Input layer mapping with activation
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for layer in self.layers:
            x = layer(x, adj)
            
        x = self.output_proj(x)
        return F.log_softmax(x, dim=1)




# ════════════════════════════════════════════════════════════════
# H2GCN — FIXED
# ════════════════════════════════════════════════════════════════

def _build_strict_khop_adj(edge_index: Tensor, num_nodes: int,
                            k: int, device) -> list:
    """Strict, disjoint r-hop adjacency matrices for r=1..k (D2)."""
    edge_index_ns, _ = remove_self_loops(edge_index)
    row, col = edge_index_ns
 
    adj_bin = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
    adj_bin[row, col] = True
    adj_bin[col, row] = True
 
    reached  = torch.eye(num_nodes, dtype=torch.bool, device=device).clone()
    frontier = adj_bin.clone()
    hop_matrices = []
 
    for r in range(1, k + 1):
        strict_hop = frontier & (~reached)
        reached    = reached | strict_hop
 
        deg     = strict_hop.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        hop_idx = strict_hop.nonzero(as_tuple=False).t()
        hop_val = (1.0 / deg[hop_idx[0]]).squeeze(-1)
 
        hop_matrices.append(SparseTensor(
            row=hop_idx[0], col=hop_idx[1], value=hop_val,
            sparse_sizes=(num_nodes, num_nodes)))
 
        if r < k:
            frontier = (frontier.float() @ adj_bin.float() > 0)
 
    return hop_matrices
 
 
class H2GCN_Net(Module):
    """
    H2GCN (Zhu et al., NeurIPS 2020) -- v4, correctly fixed dimensions.

    Architecture:
      r0 = ReLU(x @ W_embed)                              -- ego (D1)
      raw1 = [N1(r0) ; N2(r0)]              dim = hidden*k
      r1 = raw1 @ W_proj_1                  dim = hidden   (for next round's input)
      raw2 = [N1(r1) ; N2(r1)]              dim = hidden*k
      (r2 = raw2 @ W_proj_2 if more rounds follow; not needed if rounds=2)
      out = concat(r0, raw1, raw2, ...)     -- (D3) uses RAW (un-projected)
                                                hop outputs for the final
                                                classifier, preserving full
                                                information; projections are
                                                ONLY used internally to keep
                                                inter-round dimensions fixed.
    """
    def __init__(self, dataset, args):
        super().__init__()
        self.hidden  = args.hidden
        self.k       = getattr(args, 'k_hops', 2)
        self.rounds  = getattr(args, 'h2gcn_rounds', 2)
        self.dropout = args.dropout

        self.lin_in = Linear(dataset.num_features, args.hidden)

        round_dim = args.hidden * self.k   # fixed, since every round's
                                            # INPUT is always `hidden`-dim

        # Projections to bring each round's raw (hidden*k)-dim hop output
        # back down to `hidden`-dim before it becomes the NEXT round's
        # input. Only need (rounds - 1) of these (last round's output
        # isn't fed into anything further).
        self.round_proj = ModuleList([
            Linear(round_dim, args.hidden)
            for _ in range(max(self.rounds - 1, 0))
        ])

        # Final classifier: r0 (hidden) + each round's RAW hop output
        # (round_dim each) -- uses raw, un-projected outputs to avoid
        # losing information in the final representation (D3).
        total_dim = args.hidden + round_dim * self.rounds
        self.lin_out = Linear(total_dim, dataset.num_classes)

        self._adj_cache = None
        self._adj_key   = None

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        for proj in self.round_proj:
            proj.reset_parameters()
        self.lin_out.reset_parameters()

    def _get_hop_adjs(self, edge_index, num_nodes, device):
        key = (edge_index.data_ptr(), num_nodes)
        if self._adj_key != key:
            self._adj_cache = _build_strict_khop_adj(
                edge_index, num_nodes, self.k, device)
            self._adj_key = key
        return self._adj_cache

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        device    = x.device

        x  = F.dropout(x, p=self.dropout, training=self.training)
        r0 = F.relu(self.lin_in(x))     # ego embedding (D1), dim = hidden

        hop_adjs = self._get_hop_adjs(edge_index, num_nodes, device)

        round_outputs = [r0]
        r_input = r0   # always `hidden`-dim, fed into each round's hop agg

        for round_idx in range(self.rounds):
            # Hop aggregations from r_input (always hidden-dim)
            hop_terms = [hop_adj @ r_input for hop_adj in hop_adjs]
            raw_round_out = torch.cat(hop_terms, dim=-1)  # dim = hidden*k

            # Keep the RAW output for the final classifier (D3) --
            # full information preserved here.
            round_outputs.append(raw_round_out)

            # If there's a next round, project DOWN to hidden-dim so
            # the next round's hop aggregation has a consistent input
            # dimension (fixes the v3 geometric growth bug).
            if round_idx < self.rounds - 1:
                r_input = F.relu(self.round_proj[round_idx](raw_round_out))

        out = torch.cat(round_outputs, dim=-1)   # (D3): combine all rounds
        out = F.dropout(out, p=self.dropout, training=self.training)
        return F.log_softmax(self.lin_out(out), dim=-1)



# ════════════════════════════════════════════════════════════════
# ACM-GCN — FIXED
# ════════════════════════════════════════════════════════════════

def _build_norm_adj_acm(edge_index: Tensor, num_nodes: int,
                        device) -> SparseTensor:
    """Symmetric normalized adjacency with self-loops, D^{-1/2}(A+I)D^{-1/2}."""
    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index_sl, edge_weight = gcn_norm(
        edge_index_sl, add_self_loops=False, num_nodes=num_nodes)
    return SparseTensor(
        row=edge_index_sl[0], col=edge_index_sl[1], value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes))


class ACMLayer(Module):
    """
    Single ACM-GCN layer implementing the paper's exact formula (Eq. 36):

        Z^l = diag(a_I) Z^(l-1) W_I
            + diag(a_L) Â Z^(l-1) W_L
            + diag(a_H) (I - Â) Z^(l-1) W_H

    where a_I, a_L, a_H are node-wise softmax-normalized mixing weights
    learned from the three channel outputs (not from the raw input).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_I = Linear(in_dim, out_dim, bias=False)
        self.W_L = Linear(in_dim, out_dim, bias=False)
        self.W_H = Linear(in_dim, out_dim, bias=False)
        # Mixing-weight predictor: takes concat of 3 channel outputs
        self.att = Linear(out_dim * 3, 3, bias=False)

    def reset_parameters(self):
        self.W_I.reset_parameters()
        self.W_L.reset_parameters()
        self.W_H.reset_parameters()
        self.att.reset_parameters()

    def forward(self, Z: Tensor, norm_adj: SparseTensor) -> Tensor:
        N = Z.size(0)

        # Three channels, each with its OWN weight matrix (paper Eq. 36)
        Z_I = self.W_I(Z)              # identity channel
        Z_L = self.W_L(norm_adj @ Z)   # low-pass channel: Â Z W_L
        AZ  = norm_adj @ Z
        Z_H = self.W_H(Z - AZ)         # high-pass channel: (I-Â) Z W_H

        # Node-wise adaptive mixing weights (softmax-normalized, paper Sec 4)
        mix_logits = self.att(torch.cat([Z_I, Z_L, Z_H], dim=-1))  # (N,3)
        alpha = F.softmax(mix_logits, dim=-1)

        out = (alpha[:, 0:1] * Z_I
               + alpha[:, 1:2] * Z_L
               + alpha[:, 2:3] * Z_H)
        return out


class ACMGCN_Net(Module):
    """
    ACM-GCN (Luan et al., NeurIPS 2022) -- corrected, multi-layer
    implementation following the paper's exact per-layer mixing formula.
    """
    def __init__(self, dataset, args):
        super().__init__()
        self.dropout    = args.dropout
        self.num_layers = getattr(args, 'layer_num', 2)

        dims = [dataset.num_features] + [args.hidden] * self.num_layers
        self.layers = ModuleList([
            ACMLayer(dims[i], dims[i+1]) for i in range(self.num_layers)
        ])
        self.lin_out = Linear(args.hidden, dataset.num_classes)

        self._adj_cache = None
        self._adj_key   = None

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin_out.reset_parameters()

    def _get_adj(self, edge_index, num_nodes, device):
        key = (edge_index.data_ptr(), num_nodes)
        if self._adj_key != key:
            self._adj_cache = _build_norm_adj_acm(
                edge_index, num_nodes, device)
            self._adj_key = key
        return self._adj_cache

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        norm_adj = self._get_adj(edge_index, num_nodes, x.device)

        Z = x
        for i, layer in enumerate(self.layers):
            Z = F.dropout(Z, p=self.dropout, training=self.training)
            Z = layer(Z, norm_adj)
            if i < self.num_layers - 1:
                Z = F.relu(Z)

        Z = F.dropout(Z, p=self.dropout, training=self.training)
        return F.log_softmax(self.lin_out(Z), dim=-1)


# ════════════════════════════════════════════════════════════════
# FAGCN — FIXED
# ════════════════════════════════════════════════════════════════

class FAGCN_Net(Module):
    """
    FAGCN (Bo et al., AAAI 2021) -- corrected implementation.

    Key fix: the paper's core mechanism is the INITIAL RESIDUAL added at
    EVERY propagation step (Eq. 7 in the paper):

        h_i^(l+1) = eps * h_i^(0) + sum_j alpha_ij * h_j^(l)

    This residual is what prevents over-smoothing across many rounds of
    propagation -- without it (as in the buggy version, which only
    passed `x` as the static "initial" argument to FAConv without
    re-adding eps*h0 explicitly across rounds beyond what FAConv
    internally handles per-call), the eps term's effect can be diluted
    if propagation depth K is large, since PyG's FAConv only adds
    eps*x_0 ONCE per call -- which is actually correct per-call, but the
    previous version had a duplicate dead class definition and didn't
    expose eps as a tunable hyperparameter from args, risking silent
    mismatches with paper-reported configurations (paper uses eps in
    [0,1], commonly tuned per-dataset; Cornell/Texas/Wisconsin use small
    eps ~0.1-0.3, Cora/Citeseer/Pubmed use larger eps ~0.3-0.5).
    """
    def __init__(self, dataset, args):
        super().__init__()
        self.dropout = args.dropout
        self.eps     = getattr(args, 'fagcn_eps', 0.3)
        self.K       = getattr(args, 'layer_num', 2)  # propagation depth

        self.lin_in  = Linear(dataset.num_features, args.hidden)
        self.lin_out = Linear(args.hidden, dataset.num_classes)

        # One FAConv layer, reused across K propagation steps with the
        # SAME eps-weighted initial residual at every step (paper Eq. 7).
        self.prop = FAConv(args.hidden, eps=self.eps,
                           dropout=args.dropout, add_self_loops=True)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x  = F.dropout(x, p=self.dropout, training=self.training)
        h0 = F.elu(self.lin_in(x))   # initial representation h^(0)

        h = h0
        for _ in range(self.K):
            # FAConv signature: prop(x, x_0, edge_index) internally
            # computes eps*x_0 + sum_j alpha_ij * x_j, matching Eq. 7.
            h = self.prop(h, h0, edge_index)

        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.log_softmax(self.lin_out(h), dim=-1)
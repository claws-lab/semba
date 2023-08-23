import torch
from torch import nn
import math
import numpy as np
from torch.nn import functional as F
              
class Encoder(nn.Module):
    """
    Encode features to embeddings
    """

    def __init__(self, feature_dim, embed_dim, aggs, nadjs, device='cpu'):
        super(Encoder, self).__init__()

        self.device = device
        self.feat_dim = feature_dim
        self.aggs = aggs

        self.embed_dim = embed_dim
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(self.device)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.nonlinear_layer = nn.Sequential(
            nn.Linear(self.feat_dim * (nadjs + 1), self.feat_dim),
            nn.Tanh(),
            nn.Linear(self.feat_dim, self.embed_dim),

        )
        self.nonlinear_layer.apply(init_weights)


    def forward(self, x, edge_indices):
        """
        Generates embeddings for nodes.
        """
        neigh_feats = [agg.forward(x, edge_index) for edge_index, agg in zip(edge_indices, self.aggs)]
        self_feats = x
        # self_feats = self.features(torch.LongTensor(nodes).to(self.device))
        combined = torch.cat([self_feats] + neigh_feats, 1)
        combined = self.nonlinear_layer(combined)
        return combined

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape, device=indices.device)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class AttentionAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, node_num, device='cpu', dropout_rate=0.5, slope_ratio=0.1):
        super(AttentionAggregator, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))
        nn.init.kaiming_normal_(self.a.data)
        self.special_spmm = SpecialSpmm()

        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)
        self.device = device
        # self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)


    def forward(self, x, edge_index):
        new_embeddings = self.out_linear_layer(x)
        edge_h_2 = torch.cat((new_embeddings[edge_index[0]], new_embeddings[edge_index[1]]), dim=1)
        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))
        indices = edge_index
        row_sum = self.special_spmm(edge_index, 
                                    edges_h[:, 0], 
                                    torch.Size((x.shape[0], x.shape[0])), 
                                    torch.ones(size=(x.shape[0], 1)).to(self.device))
        results = self.special_spmm(edge_index, 
                                    edges_h[:, 0], 
                                    torch.Size((x.shape[0], x.shape[0])), 
                                    new_embeddings)
        output_emb = results.div(row_sum+1e-12)
        return output_emb


class SiGAT(nn.Module):

    def __init__(self, nadjs, node_feat_size, num_nodes, embedding_dim, device='cpu'):
        super(SiGAT, self).__init__()
        self.nadjs = nadjs
        aggs = [AttentionAggregator(node_feat_size, node_feat_size, num_nodes, device=device) for _ in range(nadjs)]
        self.enc = Encoder(node_feat_size, embedding_dim, aggs, nadjs, device=device)

    def forward(self, x, pos_edge_index, neg_edge_index):
        if self.nadjs == 4:
            edge_indices = [pos_edge_index, pos_edge_index[[1, 0]], neg_edge_index, neg_edge_index[[1, 0]]]
            embeds = self.enc (x, edge_indices)
        return embeds
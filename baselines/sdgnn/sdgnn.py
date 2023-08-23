import torch
from torch import nn
import math
import numpy as np
from torch.nn import functional as F

class Encoder(nn.Module):
    """
    Encode features to embeddings
    """

    def __init__(self, feature_dim, embed_dim, aggs, nadjs, device='cpu', dropout=0.5):
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
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        self.nonlinear_layer = nn.Sequential(
                nn.Linear((nadjs + 1) * feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, embed_dim)
        )

        self.nonlinear_layer.apply(init_weights)


    def forward(self, nodes):
        """
        Generates embeddings for nodes.
        """

        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()

        neigh_feats = [agg(nodes, adj, ind) for adj, agg, ind in zip(self.adj_lists, self.aggs, range(len(self.adj_lists)))]
        self_feats = self.features(torch.LongTensor(nodes).to(self.device))
        combined = torch.cat([self_feats] + neigh_feats, 1)
        combined = self.nonlinear_layer(combined)
        return combined


class AttentionAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, node_num,  dropout_rate=0.5, slope_ratio=0.1, device='cpu'):
        super(AttentionAggregator, self).__init__()

        self.device = device
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))
        nn.init.kaiming_normal_(self.a.data)

        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)


    def forward(self, nodes, adj, ind):
        """
        nodes --- list of nodes in a batch
        adj --- sp.csr_matrix
        """
        node_pku = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_pku[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))

        batch_node_num = len(unique_nodes_list)
        # this dict can map new i to originial node id
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(self.device)
        new_embeddings = self.out_linear_layer(self.features(n2))

        original_node_edge = np.array([self.unique_nodes_dict[nodes], self.unique_nodes_dict[nodes]]).T
        edges = np.vstack((edges, original_node_edge))

        edges = torch.LongTensor(edges).to(self.device)

        edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], new_embeddings[edges[:, 1], :]), dim=1)

        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))
        indices = edges
        
        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
                                         torch.Size([batch_node_num, batch_node_num]), device=self.device)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(batch_node_num, 1)).to(self.device))

        results = torch.sparse.mm(matrix, new_embeddings)

        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]

class MeanAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, node_num, device='cpu'):
        super(MeanAggregator, self).__init__()

        self.device = device
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_linear_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)

    def forward(self, nodes, adj, ind):
        """

        :param nodes:
        :param adj:
        :return:
        """
        mask = [1, 1, 0, 0]
        node_tmp = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_tmp[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))
        batch_node_num = len(unique_nodes_list)
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        ## transform 2 new axis
        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(self.device)
        new_embeddings = self.out_linear_layer(self.features(n2))
        edges = torch.LongTensor(edges).to(self.device)

        values = torch.where(edges[:, 0] == edges[:, 1], torch.FloatTensor([mask[ind]]).to(self.device), torch.FloatTensor([1]).to(self.device))
        # values = torch.ones(edges.shape[0]).to(self.device)
        matrix = torch.sparse_coo_tensor(edges.t(), values, torch.Size([batch_node_num, batch_node_num]), device=self.device)
        row_sum = torch.spmm(matrix, torch.ones(size=(batch_node_num, 1)).to(self.device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(self.device), row_sum)

        results = torch.spmm(matrix, new_embeddings)
        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]



class SDGNN(nn.Module):

    def __init__(self, enc, embedding_size=32, device='cpu'):
        super(SDGNN, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.enc = enc
        self.score_function1 = nn.Sequential(
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )
        self.score_function2 = nn.Sequential(
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(self.embedding_size * 2, 1)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds

    def criterion(self, nodes, pos_neighbors, neg_neighbors, adj_lists1_1, adj_lists2_1, weight_dict):
        pos_neighbors_list = [set.union(pos_neighbors[i]) for i in nodes]
        neg_neighbors_list = [set.union(neg_neighbors[i]) for i in nodes]
        unique_nodes_list = list(set.union(*pos_neighbors_list).union(*neg_neighbors_list).union(nodes))
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}
        nodes_embs = self.enc(unique_nodes_list)

        loss_total = 0
        for index, node in enumerate(nodes):
            z1 = nodes_embs[unique_nodes_dict[node], :]
            pos_neigs = list([unique_nodes_dict[i] for i in pos_neighbors[node]])
            neg_neigs = list([unique_nodes_dict[i] for i in neg_neighbors[node]])
            pos_num = len(pos_neigs)
            neg_num = len(neg_neigs)

            sta_pos_neighs = list([unique_nodes_dict[i] for i in adj_lists1_1[node]])
            sta_neg_neighs = list([unique_nodes_dict[i] for i in adj_lists2_1[node]])

            pos_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists1_1[node]]).to(self.device)
            neg_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists2_1[node]]).to(self.device)

            if pos_num > 0:
                pos_neig_embs = nodes_embs[pos_neigs, :]
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [pos_neig_embs, z1]),
                                                              torch.ones(pos_num).to(self.device))

                if len(sta_pos_neighs) > 0:
                    sta_pos_neig_embs = nodes_embs[sta_pos_neighs, :]

                    z11 = z1.repeat(len(sta_pos_neighs), 1)
                    rs = self.fc(torch.cat([z11, sta_pos_neig_embs], 1)).squeeze(-1)
                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.ones(len(sta_pos_neighs)).to(self.device), \
                                                                   weight=pos_neigs_weight
                                                                   )
                    s1 = self.score_function1(z1).repeat(len(sta_pos_neighs), 1)
                    s2 = self.score_function2(sta_pos_neig_embs)

                    q = torch.where((s1 - s2) > -0.5, torch.Tensor([-0.5]).repeat(s1.shape).to(self.device), s1 - s2)
                    tmp = (q - (s1 - s2))
                    loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])

                loss_total += loss_pku

            if neg_num > 0:
                neg_neig_embs = nodes_embs[neg_neigs, :]
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [neg_neig_embs, z1]),
                                                              torch.zeros(neg_num).to(self.device))
                if len(sta_neg_neighs) > 0:
                    sta_neg_neig_embs = nodes_embs[sta_neg_neighs, :]

                    z12 = z1.repeat(len(sta_neg_neighs), 1)
                    rs = self.fc(torch.cat([z12, sta_neg_neig_embs], 1)).squeeze(-1)

                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.zeros(len(sta_neg_neighs)).to(self.device), \
                                                                   weight=neg_neigs_weight)

                    s1 = self.score_function1(z1).repeat(len(sta_neg_neighs), 1)
                    s2 = self.score_function2(sta_neg_neig_embs)

                    q = torch.where(s1 - s2 > 0.5, s1 - s2, torch.Tensor([0.5]).repeat(s1.shape).to(self.device))

                    tmp = (q - (s1 - s2))
                    loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])

                loss_total += loss_pku

        return loss_total


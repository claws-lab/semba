import torch
import math
import numpy as np
from torch.nn import functional as F

from baselines.caw.caw import CAWN
from baselines.sdgnn.sdgnn import SDGNN
from baselines.sgcn.sgcn import SignedGCN
from baselines.sigat.sigat import SiGAT
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)

from baselines.tgn.tgn import TGNMemory, LastNeighborLoader, IdentityMessage, MeanAggregator
from src.semba import SEMBA, TimeEncoder
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv, SignedConv
from torch.nn import Linear, ReLU, LSTM
from torch_geometric.nn import GCNConv
import torch_geometric as tg

class GraphAttentionEmbedding(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, dropout=0.0):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        heads = 8
        self.conv = TransformerConv(in_channels, out_channels // heads, heads=heads,
                                    dropout=dropout, edge_dim=edge_dim)

    def forward(self, feats, x, last_update, edge_index, t, msg):

        if last_update is not None:
            last_update = torch.max(last_update, 1)[0] if last_update.ndim == 2 else last_update
            rel_t = last_update[edge_index[0]] - t
        else:
            rel_t = t
        rel_t_enc = self.time_enc(rel_t.to(feats.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        # print (edge_index.max(), edge_attr.max())
     
        return self.conv(torch.cat([x, feats], dim=-1) if x is not None else feats, edge_index, edge_attr)

def pos_embedding_loss(z, pos_edge_index):
    i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
    out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
    return torch.clamp(out, min=0).mean()

def neg_embedding_loss(z, neg_edge_index):
    i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
    out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
    return torch.clamp(out, min=0).mean()

                    
class STGNN (torch.nn.Module):
    def __init__ (
        self, 
        model_name, 
        task,
        num_feats,
        num_nodes,
        embedding_dim,
        num_layers,
        ablation='smem-ba-prop',
        nwts=1,
        memory_dim=64,
        time_dim=32,
        dropout=0.5,
        nbr_size=10,
        device='cpu',
        debug=False,
    ):
        self.model_name = model_name
        self.task = task
        self.device = device
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.debug = debug
        super(STGNN, self).__init__()
        self.seen_pos_ei = torch.empty(2, 0, dtype=torch.long, device=device)
        self.seen_neg_ei = torch.empty(2, 0, dtype=torch.long, device=device)
        self.seen_pos_times = torch.empty(0, dtype=torch.long, device=device)
        self.seen_neg_times = torch.empty(0, dtype=torch.long, device=device)
        self.seen_pos_weights = torch.empty(0, dtype=torch.long, device=device)
        self.seen_neg_weights = torch.empty(0, dtype=torch.long, device=device)
        self.seen_times = torch.cat([self.seen_pos_times, self.seen_neg_times], dim=0)
        self.seen_weights = torch.cat([self.seen_pos_weights, self.seen_neg_weights], dim=0)
        self.seen_ei = torch.cat([self.seen_pos_ei, self.seen_neg_ei], dim=1)
        self.disc_timecuts = [0]
        if model_name == 'sgcn':
            if self.debug:
                print (num_feats, embedding_dim, num_layers)
            self.model = SignedGCN(num_feats, embedding_dim, num_layers=num_layers, device=device).to(device=device)
        elif model_name == 'gcn':
            self.model = tg.nn.Sequential('x, edge_index', [
                (GCNConv(num_feats, embedding_dim), 'x, edge_index -> x'),
                ReLU(inplace=True),
                (GCNConv(embedding_dim, embedding_dim), 'x, edge_index -> x'),
                ReLU(inplace=True),
            ])
        elif model_name == 'sgclstm':
            self.gmodel = SignedGCN(num_feats, embedding_dim, num_layers=num_layers, device=device).to(device=device)
            self.tmodel = LSTM(embedding_dim, embedding_dim).to(device=device)
            self.model = torch.nn.Sequential(self.gmodel, self.tmodel)
        elif model_name == 'sigat':
            self.model = SiGAT(4, num_feats, num_nodes, embedding_dim, device=device).to(device=device)
        elif model_name == 'sdgnn':
            self.model = SDGNN(4, num_feats, num_nodes, embedding_dim, device=device).to(device=device)
        elif model_name == 'tgn':
            self.mem_model = TGNMemory (num_nodes=num_nodes, raw_msg_dim=nwts, 
                                        memory_dim=memory_dim, time_dim=time_dim,
                                        message_module=IdentityMessage(nwts, memory_dim, time_dim),
                                        aggregator_module=MeanAggregator(), 
                                        device=device).to(device)
            self.nbrLoader = LastNeighborLoader(num_nodes, size=10, device=device)
            self.mem2emb = GraphAttentionEmbedding(in_channels=memory_dim+num_feats,
                                                    out_channels=embedding_dim,
                                                    msg_dim=nwts,
                                                    time_enc=self.mem_model.time_enc,
                                                    dropout=dropout).to(device)
            self.model = torch.nn.Sequential(self.mem_model, self.mem2emb)
        elif model_name == 'semba':
            self.mem_model = SEMBA (num_nodes, nwts, memory_dim, time_dim,
                                    message_module_pos=IdentityMessage(nwts, memory_dim, time_dim),
                                    message_module_neg=IdentityMessage(nwts, memory_dim, time_dim),
                                    aggregator_module_pos=MeanAggregator(),
                                    aggregator_module_neg=MeanAggregator(),
                                    device=device).to(device)
            self.nbrLoaderPos = LastNeighborLoader(num_nodes, size=nbr_size, device=device)
            self.nbrLoaderNeg = LastNeighborLoader(num_nodes, size=nbr_size, device=device)
            self.mem2emb = GraphAttentionEmbedding(in_channels=2*memory_dim+num_feats,
                                                    out_channels=embedding_dim,
                                                    msg_dim=nwts,
                                                    time_enc=self.mem_model.time_enc,
                                                    dropout=dropout).to(device)
            self.model = torch.nn.Sequential(self.mem_model, self.mem2emb)
        elif model_name == 'semba-noprop':
            memory_dim = embedding_dim//2
            self.model = SEMBA (num_nodes, nwts, memory_dim, time_dim,
                                message_module_pos=IdentityMessage(nwts, memory_dim, time_dim),
                                message_module_neg=IdentityMessage(nwts, memory_dim, time_dim),
                                aggregator_module_pos=MeanAggregator(),
                                aggregator_module_neg=MeanAggregator(),
                                device=device).to(device)
        elif model_name == 'semba-noba':
            self.mem_model = SEMBA (num_nodes, nwts, memory_dim, time_dim,
                                    message_module_pos=IdentityMessage(nwts, memory_dim, time_dim),
                                    message_module_neg=IdentityMessage(nwts, memory_dim, time_dim),
                                    aggregator_module_pos=MeanAggregator(),
                                    aggregator_module_neg=MeanAggregator(),
                                    balanced_aggr=False,
                                    device=device).to(device)
            self.nbrLoaderPos = LastNeighborLoader(num_nodes, size=nbr_size, device=device)
            self.nbrLoaderNeg = LastNeighborLoader(num_nodes, size=nbr_size, device=device)
            self.mem2emb = GraphAttentionEmbedding(in_channels=2*memory_dim+num_feats,
                                                    out_channels=embedding_dim,
                                                    msg_dim=nwts,
                                                    time_enc=self.mem_model.time_enc,
                                                    dropout=dropout).to(device)
            self.model = torch.nn.Sequential(self.mem_model, self.mem2emb)
        elif model_name == 'prop':
            print (num_feats, embedding_dim, nwts)
            self.model = GraphAttentionEmbedding(in_channels=num_feats,
                                                    out_channels=embedding_dim,
                                                    msg_dim=nwts,
                                                    time_enc=TimeEncoder(time_dim),
                                                    dropout=dropout).to(device)
        elif model_name == 'caw':
            self.model = CAWN(num_feats, embedding_dim, num_layers=2, lamb=0, device=device).to(device=device)

        if task == 'signlink_class':
            self.lin = torch.nn.Linear(2 * embedding_dim, 3).to(device=device)
        elif task == 'sign_class':
            self.lin = torch.nn.Linear(2 * embedding_dim, 1).to(device=device)
        elif task == 'link_pred':
            self.lin = torch.nn.Linear(2 * embedding_dim, 1).to(device=device)
        elif task == 'signwt_pred':
            self.lin = torch.nn.Linear(2 * embedding_dim, 1).to(device=device)

    def forward (self, x, pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights, to_update=False):
        # inductive setting so edge_indices not included
        if self.model_name in ['sgcn', 'sigat']:
            try:
                z = self.model(x, self.seen_pos_ei, self.seen_neg_ei)
            except:
                z = torch.rand(x.shape[0], self.embedding_dim).to(x.device)
            if to_update:
                self.update_memory(pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights)
        if self.model_name in ['gcn']:
            try:
                z = self.model(x, self.seen_ei)
            except:
                z = torch.rand(x.shape[0], self.embedding_dim).to(x.device)
            if to_update:
                self.update_memory(pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights)
        elif self.model_name in ['sgclstm']:
            try:
                seen_pos_ei_disc, seen_neg_ei_disc = self.get_disc_history_ei()
                z = self.tmodel(torch.stack([self.gmodel(x, pei, nei) 
                                for pei, nei in zip(seen_pos_ei_disc, seen_neg_ei_disc)]))[0][0]
            except:
                z = torch.rand(x.shape[0], self.embedding_dim).to(x.device)
            if to_update:
                self.update_memory(pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights)
        elif self.model_name == 'tgn':
            # self.nbrLoader.insert(edge_index[0], edge_index[1], times, weights)
            # n_id = edge_index.unique()
            n_id = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            z, last_update = self.mem_model(n_id)
            # emb
            if to_update:
                self.update_memory(pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights)
            # n_id, smpld_ei, smpld_times, smpld_weights = self.nbrLoader(n_id)
            z = self.mem2emb(x, z, last_update, self.seen_ei, self.seen_times, self.seen_weights)
            # self.mem_model.update_state(pos_edge_index[0], pos_edge_index[1], pos_times, 1*pos_weights)
            # self.mem_model.update_state(neg_edge_index[0], neg_edge_index[1], neg_times, -1*neg_weights)
        elif self.model_name == 'semba':
            n_id = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            z, last_update = self.mem_model(n_id, n_id)
            # embs
            if to_update:
                self.update_memory(pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights)
            z = self.mem2emb(x, z, last_update, self.seen_ei, self.seen_times, self.seen_weights)
        elif self.model_name == 'semba-noprop':
            n_id = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            z, last_update = self.model(n_id, n_id)
            # embs
            if to_update:
                self.update_memory(pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights)
        elif self.model_name == 'prop':
            # embs
            try:
                z = self.model(x, None, None, self.seen_ei, self.seen_times, self.seen_weights)
            except:
                z = torch.rand(x.shape[0], self.embedding_dim).to(x.device)
            if to_update:
                self.update_memory(pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights)
        elif self.model_name == 'smem':
            n_id = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            z, _ = self.mem_model(n_id, n_id)
        elif self.model_name == 'caw':
            return self.model()
            # CAWN(num_feats, embedding_dim, num_layers=2, lamb=0, device=device).to(device=device)
        return z

    def get_disc_history_ei (self):
        seen_pos_ei_disc =  [self.seen_pos_ei[:, self.disc_timecuts[tind-1]:self.disc_timecuts[tind]] 
                                        for tind in range(1, len(self.disc_timecuts))]
        seen_neg_ei_disc =  [self.seen_neg_ei[:, self.disc_timecuts[tind-1]:self.disc_timecuts[tind]] 
                                        for tind in range(1, len(self.disc_timecuts))]
        return seen_pos_ei_disc, seen_neg_ei_disc
    
    def update_memory (self, pos_edge_index, neg_edge_index, pos_times, neg_times, pos_weights, neg_weights):
        self.seen_pos_ei = torch.cat((self.seen_pos_ei, pos_edge_index), dim=-1) if self.seen_pos_ei.shape[1] != 0 else pos_edge_index
        self.seen_neg_ei = torch.cat((self.seen_neg_ei, neg_edge_index), dim=-1) if self.seen_neg_ei.shape[1] != 0 else neg_edge_index
        self.seen_pos_times = torch.cat((self.seen_pos_times, pos_times), dim=0) if self.seen_pos_times.shape[0] != 0 else pos_times
        self.seen_neg_times = torch.cat((self.seen_neg_times, neg_times), dim=0) if self.seen_neg_times.shape[0] != 0 else neg_times
        self.seen_pos_weights = torch.cat((self.seen_pos_weights, pos_weights), dim=0) if self.seen_pos_weights.shape[0] != 0 else pos_weights
        self.seen_neg_weights = torch.cat((self.seen_neg_weights, neg_weights), dim=0) if self.seen_neg_weights.shape[0] != 0 else neg_weights
        self.seen_times = torch.cat([self.seen_pos_times, self.seen_neg_times], dim=0)
        self.seen_weights = torch.cat([self.seen_pos_weights, self.seen_neg_weights], dim=0)
        self.seen_ei = torch.cat([self.seen_pos_ei, self.seen_neg_ei], dim=1)
        self.disc_timecuts.append(len(self.seen_pos_times))
        if self.model_name == 'tgn':
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            times = torch.cat([pos_times, neg_times], dim=-1)
            weights = torch.cat([pos_weights, neg_weights], dim=0)
            self.mem_model.update_state(edge_index[0], edge_index[1], times, weights)
        elif self.model_name == 'semba':
            self.mem_model.update_state(pos_edge_index[0], neg_edge_index[0], pos_edge_index[1], 
                                        neg_edge_index[1], pos_times, neg_times, pos_weights, neg_weights)
        elif self.model_name == 'semba-noprop':
            self.model.update_state(pos_edge_index[0], neg_edge_index[0], pos_edge_index[1], 
                                    neg_edge_index[1], pos_times, neg_times, pos_weights, neg_weights)

    def reset_memory (self):
        self.seen_pos_ei = torch.empty(2, 0, dtype=torch.long, device=self.device)
        self.seen_neg_ei = torch.empty(2, 0, dtype=torch.long, device=self.device)
        self.seen_pos_times = torch.empty(0, dtype=torch.long, device=self.device)
        self.seen_neg_times = torch.empty(0, dtype=torch.long, device=self.device)
        self.seen_pos_weights = torch.empty(0, dtype=torch.long, device=self.device)
        self.seen_neg_weights = torch.empty(0, dtype=torch.long, device=self.device)
        self.seen_times = torch.cat([self.seen_pos_times, self.seen_neg_times], dim=0)
        self.seen_weights = torch.cat([self.seen_pos_weights, self.seen_neg_weights], dim=0)
        self.seen_ei = torch.cat([self.seen_pos_ei, self.seen_neg_ei], dim=1)
        self.disc_timecuts = [0]
        if self.model_name in ['tgn', 'semba']:
            self.mem_model.reset_state()
        elif self.model_name == 'semba-noprop':
            self.model.reset_state()


    def discriminate(self, z, edge_index):
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(self, z, pos_edge_index, neg_edge_index, none_edge_index=None, neg_wt=1.0, null_wt=1.0, null_nsamples=10):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        if none_edge_index is None:
            none_edge_index = negative_sampling(edge_index, num_nodes=z.size(0), 
                                                num_neg_samples=null_nsamples*edge_index.shape[1]
                                            ).to(device=self.device)
            # none_edge_index = negative_sampling(edge_index, z.size(0), null_nsamples).to(device=self.device)
        nll_loss = 0
        nll_loss += F.nll_loss(
            torch.log(self.predict(z, pos_edge_index[0], pos_edge_index[1])),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0), 
            weight=None)
        nll_loss += neg_wt * F.nll_loss(
            torch.log(self.predict(z, neg_edge_index[0], neg_edge_index[1])),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1),
            weight=None)
        nll_loss += null_wt * F.nll_loss(
            torch.log(self.predict(z, none_edge_index[0], none_edge_index[1])),
            none_edge_index.new_full((none_edge_index.size(1), ), 2),
            weight=None)
        return nll_loss / 3.0

    def predict(self, z, src, dst, classify=False):
        emb_nodepair = torch.cat((z[src], z[dst]), dim=1)
        value = self.lin(emb_nodepair)
        if self.task == "signlink_class":
            return (torch.softmax(value, dim=1) if not classify else value.argmax(dim=1).squeeze())
        elif self.task in ["sign_class", "link_pred"]:
            return (torch.sigmoid(value) if not classify else torch.where(value>0.5, 1., 0.)).squeeze()
        elif self.task == "signwt_pred":
            return value.squeeze()

    def loss(self, z, pos_edge_index, neg_edge_index, null_edge_index=None, signed_edge_weights=None, 
             neg_wt=1.0, null_wt=1.0, null_nsamples=10, lambda_struc=0, lambda_l1=0, lambda_l2=0):
        if self.task == "signlink_class":
            loss = self.nll_loss(z, pos_edge_index, neg_edge_index, none_edge_index=null_edge_index,
                                 neg_wt=neg_wt, null_wt=null_wt, null_nsamples=null_nsamples)
        elif self.task == "sign_class":
            pos_proba = self.predict(z, pos_edge_index[0], pos_edge_index[1])
            loss = F.binary_cross_entropy(pos_proba, torch.ones_like (pos_proba))
            if neg_edge_index.shape[1] != 0:
                neg_proba = self.predict(z, neg_edge_index[0], neg_edge_index[1])
                loss += neg_wt * F.binary_cross_entropy(neg_proba, torch.zeros_like (neg_proba))
        elif self.task == "link_pred":
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_proba = self.predict(z, edge_index[0], edge_index[1])
            loss = F.binary_cross_entropy(edge_proba, torch.ones_like (edge_proba))
            if null_edge_index.shape[1] != 0:
                noedge_proba = self.predict(z, null_edge_index[0], null_edge_index[1])
                loss += null_wt * F.binary_cross_entropy(noedge_proba, torch.zeros_like (noedge_proba))
        elif self.task == "signwt_pred":
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            pred_sign_wt = self.predict(z, edge_index[0], edge_index[1])
            loss = F.mse_loss(pred_sign_wt, signed_edge_weights)
        if lambda_struc != 0:
            loss_posStruc = pos_embedding_loss(z, pos_edge_index)
            loss_negStruc = neg_embedding_loss(z, neg_edge_index)
            loss += lambda_struc * (loss_posStruc + loss_negStruc)
        all_model_params = torch.cat([x.view(-1) for x in self.model.parameters()])
        loss += lambda_l1 * all_model_params.norm(p=1) + lambda_l2 * all_model_params.norm(p=2)
        return loss

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self.model.train()
            self.lin.train()
        super(STGNN, self).train(mode)

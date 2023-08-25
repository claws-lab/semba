import copy
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, GRUCell
from torch_scatter import scatter, scatter_max, scatter_mean

from torch_geometric.nn.inits import zeros

class SEMBA(torch.nn.Module):

    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module_pos: Callable,
                 message_module_neg: Callable,
                 aggregator_module_pos: Callable, aggregator_module_neg: Callable, 
                 balanced_aggr=True,
                 device='cpu'):
        
        super(SEMBA, self).__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.time_enc = TimeEncoder(time_dim)
        
        # create a pos and neg versions of message module, aggregator and memory updater
        
        self.msg_s_module_pos, self.msg_s_module_neg = message_module_pos, message_module_neg 
        self.msg_d_module_pos, self.msg_d_module_neg = copy.deepcopy(message_module_pos), copy.deepcopy(message_module_neg)
        self.aggr_module_pos, self.aggr_module_neg = aggregator_module_pos, aggregator_module_neg
        self.gru_pos = GRUCell(message_module_pos.out_channels, memory_dim)
        self.gru_neg = GRUCell(message_module_neg.out_channels, memory_dim)

        self.register_buffer('memory_pos', torch.empty(num_nodes, memory_dim))
        self.register_buffer('memory_neg', torch.empty(num_nodes, memory_dim))
        
        last_update_pos = torch.empty(self.num_nodes, dtype=torch.long)
        last_update_neg = torch.empty(self.num_nodes, dtype=torch.long)
        
        self.register_buffer('last_update_pos', last_update_pos)
        self.register_buffer('last_update_neg', last_update_neg)
        
        self.register_buffer('__assoc__',
                             torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer('__assocPos__',
                             torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer('__assocNeg__',
                             torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store_pos, self.msg_s_store_neg = {}, {}
        self.msg_d_store_pos, self.msg_d_store_neg = {}, {}

        self.device = device
        self.balanced_aggr = balanced_aggr

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.msg_s_module_pos, 'reset_parameters'):
            self.msg_s_module_pos.reset_parameters()
        if hasattr(self.msg_s_module_neg, 'reset_parameters'):
            self.msg_s_module_neg.reset_parameters()
            
        if hasattr(self.msg_d_module_pos, 'reset_parameters'):
            self.msg_d_module_pos.reset_parameters()
        if hasattr(self.msg_d_module_neg, 'reset_parameters'):
            self.msg_d_module_neg.reset_parameters()    
           
        if hasattr(self.aggr_module_pos, 'reset_parameters'):
            self.aggr_module_pos.reset_parameters()
        if hasattr(self.aggr_module_neg, 'reset_parameters'):
            self.aggr_module_neg.reset_parameters()
            
        self.time_enc.reset_parameters()
        self.gru_pos.reset_parameters()
        self.gru_neg.reset_parameters()
        self.reset_state()

    def init_memory(self, x):
        self.memory = x

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory_pos)
        zeros(self.memory_neg)
        
        zeros(self.last_update_pos)
        zeros(self.last_update_neg)
        self.__reset_message_store__()

    def detach(self):
        """Detachs the memory from gradient computation."""
        self.memory_pos.detach_()
        self.memory_neg.detach_()

    def forward(self, n_id_pos: Tensor, n_id_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        
        # if torch.all(self.memory == 0):
        #     self.init_memory(x)

        n_id = torch.cat([n_id_pos, n_id_neg]).unique()
        # get pos and neg memories for pos and neg nodes respectively
        if self.training:
            memory_pos, memory_neg, last_update_pos, last_update_neg = self.__get_updated_memory__(n_id_pos, n_id_neg)    
            # memory_pos, last_update_pos, last_update_neg = self.__get_updated_memory__(n_id_pos, n_id_neg)    
            
            # self.memory_pos[n_id], self.memory_neg[n_id] = memory_pos, memory_neg
            # self.last_update_pos[n_id_pos], self.last_update_neg[n_id_neg] = last_update_pos, last_update_neg
            
            memory = torch.cat([memory_pos[n_id], memory_neg[n_id]], -1)
            last_update = torch.stack([last_update_pos[n_id], last_update_neg[n_id]], -1)
                        
        else:
            
            memory = torch.cat([self.memory_pos[n_id], self.memory_neg[n_id]], -1)
            last_update = torch.stack([self.last_update_pos[n_id], self.last_update_neg[n_id]], -1)
            
        return memory, last_update

    def update_state(self, src_pos, src_neg, dst_pos, dst_neg, t_pos, t_neg, msg_pos, msg_neg):
        
        """Updates the memory with newly encountered interactions"""
        n_id_pos = torch.cat([src_pos, dst_pos]).unique()
        n_id_neg = torch.cat([src_neg, dst_neg]).unique()

        if self.training:
            self.__update_memory__(n_id_pos, n_id_neg)
            self.__update_msg_store__(src_pos, dst_pos, t_pos, msg_pos, self.msg_s_store_pos)
            self.__update_msg_store__(src_neg, dst_neg, t_neg, msg_neg, self.msg_s_store_neg)
            self.__update_msg_store__(dst_pos, src_pos, t_pos, msg_pos, self.msg_d_store_pos)
            self.__update_msg_store__(dst_neg, src_neg, t_neg, msg_neg, self.msg_d_store_neg)
        
        else:
            self.__update_msg_store__(src_pos, dst_pos, t_pos, msg_pos, self.msg_s_store_pos)
            self.__update_msg_store__(src_neg, dst_neg, t_neg, msg_neg, self.msg_s_store_neg)
            self.__update_msg_store__(dst_pos, src_pos, t_pos, msg_pos, self.msg_d_store_pos)
            self.__update_msg_store__(dst_neg, src_neg, t_neg, msg_neg, self.msg_d_store_neg)
            self.__update_memory__(n_id_pos, n_id_neg)

    def __reset_message_store__(self):
        i = self.memory_pos.new_empty((0, ), dtype=torch.long, device=self.device)
        k = self.memory_neg.new_empty((0, ), dtype=torch.long, device=self.device)
        
        msg_pos = self.memory_pos.new_empty((0, self.raw_msg_dim), device=self.device)
        msg_neg = self.memory_neg.new_empty((0, self.raw_msg_dim), device=self.device)
        
        # Message store format: (src, dst, t, msg)
        self.msg_s_store_pos = {j: (i, i, i, msg_pos) for j in range(self.num_nodes)}
        self.msg_s_store_neg = {j: (k, k, k, msg_neg) for j in range(self.num_nodes)}
        
        self.msg_d_store_pos = {j: (i, i, i, msg_pos) for j in range(self.num_nodes)}
        self.msg_d_store_neg = {j: (k, k, k, msg_neg) for j in range(self.num_nodes)}

    def __update_memory__(self, n_id_pos, n_id_neg):
        n_id = torch.cat([n_id_pos, n_id_neg]).unique()
        memory_pos, memory_neg, last_update_pos, last_update_neg = self.__get_updated_memory__(n_id_pos, n_id_neg)
        # memory_pos, last_update_pos, last_update_neg = self.__get_updated_memory__(n_id_pos, n_id_neg)
        
        self.memory_pos[n_id], self.memory_neg[n_id] = memory_pos, memory_neg
        # self.memory_pos[n_id] = memory_pos
        self.last_update_pos[n_id_pos], self.last_update_neg[n_id_neg] = last_update_pos, last_update_neg

    def __get_updated_memory__(self, n_id_pos, n_id_neg):
        n_id = torch.cat([n_id_pos, n_id_neg]).unique()
        # self.__assocPos__[n_id_pos] = torch.arange(n_id_pos.size(0), device=n_id_pos.device)
        # self.__assocNeg__[n_id_neg] = torch.arange(n_id_neg.size(0), device=n_id_neg.device)
        self.__assocPos__[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        self.__assocNeg__[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Equation 1 -> m_i+   (src -> dst)
        msg_s_pos, t_s_pos, src_s_pos, dst_s_pos = self.__compute_msgPos__(n_id_pos, n_id_neg, 
                                                        self.msg_s_store_pos, self.msg_s_store_neg,
                                                        self.msg_s_module_pos, self.msg_s_module_neg)
        
        # Equation 2 -> m_i-   (src -> dst)
        msg_s_neg, t_s_neg, src_s_neg, dst_s_neg = self.__compute_msgNeg__(n_id_pos, n_id_neg, 
                                                        self.msg_s_store_neg, self.msg_s_store_neg,
                                                        self.msg_s_module_pos, self.msg_s_module_neg)
        
        # Equation 3 -> m_j+   (dst -> src)
        msg_d_pos, t_d_pos, src_d_pos, dst_d_pos = self.__compute_msgPos__(n_id_pos, n_id_neg, 
                                                        self.msg_d_store_pos, self.msg_d_store_neg,
                                                        self.msg_d_module_pos, self.msg_d_module_neg)
        
        # Equation 4 -> m_j-   (dst -> src)
        msg_d_neg, t_d_neg, src_d_neg, dst_d_neg = self.__compute_msgNeg__(n_id_pos, n_id_neg, 
                                                        self.msg_d_store_neg, self.msg_d_store_neg,
                                                        self.msg_d_module_pos, self.msg_d_module_neg)

        # Aggregate messages.
        idx_pos = torch.cat([src_s_pos, src_d_pos], dim=0)
        idx_neg = torch.cat([src_s_neg, src_d_neg], dim=0)
            
        msg_pos = torch.cat([msg_s_pos, msg_d_pos], dim=0).data.clone()
        msg_neg = torch.cat([msg_s_neg, msg_d_neg], dim=0).data.clone()
        
        t_pos = torch.cat([t_s_pos, t_d_pos], dim=0)
        t_neg = torch.cat([t_s_neg, t_d_neg], dim=0)
        
        # print (msg_pos.shape, self.__assocPos__[idx_pos], t_pos, n_id_pos.size(0))
        # print (msg_neg.shape, self.__assocNeg__[idx_neg], t_neg, n_id_neg.size(0))
        # self.__assocNeg__[n_id_neg] = torch.arange(n_id_neg.size(0), device=n_id_neg.device)
        # print (msg_neg.shape, self.__assocNeg__[idx_neg], t_neg, n_id_neg.size(0))
        aggr_pos = self.aggr_module_pos(msg_pos, self.__assocPos__[idx_pos], t_pos, n_id.size(0))
        aggr_neg = self.aggr_module_neg(msg_neg, self.__assocNeg__[idx_neg], t_neg, n_id.size(0))
        
        # Get local copy of updated memory
        mems = self.memory_pos.data.clone()[n_id]
        aggr_pos = aggr_pos
        memory_pos = self.gru_pos(aggr_pos, mems)
        mems = self.memory_neg.data.clone()[n_id]
        aggr_neg = aggr_neg
        memory_neg = self.gru_neg(aggr_neg, mems) 

        # Get local copy of updated `last_update`.
        dim_size = self.last_update_pos.size(0)
        last_update_pos = scatter_max(t_pos, idx_pos, dim=0, dim_size=dim_size)[0][n_id_pos]
        
        dim_size = self.last_update_neg.size(0)
        last_update_neg = scatter_max(t_neg, idx_neg, dim=0, dim_size=dim_size)[0][n_id_neg]

        # return memory_pos, last_update_pos, last_update_neg # memory_neg, last_update_pos, last_update_neg
        return memory_pos, memory_neg, last_update_pos, last_update_neg
           
    def __update_msg_store__(self, src, dst, t, raw_msg, msg_store):
        
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def __compute_msgPos__(self, n_id_pos, n_id_neg, msg_store_pos, msg_store_neg, msg_module_pos, msg_module_neg):
        if n_id_pos.size(0):
            m_friends, t_f, src_f, dst_f = self.__msg_helper__(n_id_pos, msg_store_pos, msg_module_pos, 
                                                        self.memory_pos, self.memory_pos, self.last_update_pos)
            if not n_id_neg.size(0):
                return m_friends, t_f, src_f, dst_f
        if n_id_neg.size(0):
            m_enemies, t_e, src_e, dst_e = self.__msg_helper__(n_id_neg, msg_store_neg, msg_module_neg, 
                                                        self.memory_pos, self.memory_neg, self.last_update_neg)
            if not n_id_pos.size(0):
                return m_enemies, t_e, src_e, dst_e
        msg = torch.cat([m_friends, m_enemies], dim=0)
        t = torch.cat([t_f, t_e], dim=0)
        src = torch.cat([src_f, src_e], dim=0)
        dst = torch.cat([dst_f, dst_e], dim=0)
        return msg, t, src, dst

    def __compute_msgNeg__(self, n_id_pos, n_id_neg, msg_store_pos, msg_store_neg, msg_module_pos, msg_module_neg):
        if n_id_neg.size(0):
            m_friends, t_f, src_f, dst_f = self.__msg_helper__(n_id_neg, msg_store_neg, msg_module_neg, 
                                                        self.memory_neg, self.memory_neg, self.last_update_neg)
            if not n_id_pos.size(0):
                return m_friends, t_f, src_f, dst_f
        if n_id_pos.size(0):
            m_enemies, t_e, src_e, dst_e = self.__msg_helper__(n_id_pos, msg_store_pos, msg_module_pos, 
                                                        self.memory_neg, self.memory_pos, self.last_update_pos)
            if not n_id_neg.size(0):
                return m_enemies, t_e, src_e, dst_e
        msg = torch.cat([m_friends, m_enemies], dim=0)
        t = torch.cat([t_f, t_e], dim=0)
        src = torch.cat([src_f, src_e], dim=0)
        dst = torch.cat([dst_f, dst_e], dim=0)
        return msg, t, src, dst


    def __compute_msg__(self, n_id_pos, n_id_neg, msg_store, msg_module, memory_one, memory_two):
        
        # Compute messages for equation 1 - 4
        
        # case e_ij = +1 
        m_friends, t_f, src_f, dst_f = self.__msg_helper__(
            n_id_pos, msg_store, msg_module, memory_one, memory_one, self.last_update_pos)
        
        # case e_ij = -1
        if n_id_neg.size(0):  # Compute them only if there are negative edges in the batch
            m_enemies, t_e, src_e, dst_e = self.__msg_helper__(
                n_id_neg, msg_store, msg_module, memory_one, memory_two, self.last_update_neg)

        # Stack 
            msg = torch.cat([m_friends, m_enemies], dim=0)
            t = torch.cat([t_f, t_e], dim=0)
            src = torch.cat([src_f, src_e], dim=0)
            dst = torch.cat([dst_f, dst_e], dim=0)
            
        else:
            msg = m_friends.to(device=n_id_pos.device)
            t = t_f.to(device=n_id_pos.device)
            src = src_f.to(device=n_id_pos.device)
            dst = dst_f.to(device=n_id_pos.device) 
        
        return msg, t, src, dst

    
    def __msg_helper__(self, n_id, msg_store, msg_module, memory_s, memory_d, last_update):
        
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(memory_s[src], memory_d[dst], raw_msg, t_enc)

        return msg, t, src, dst
    
    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self.__update_memory__(
                torch.arange(self.num_nodes, device=self.memory_pos.device), 
                torch.arange(self.num_nodes, device=self.memory_pos.device))
            self.__reset_message_store__()
        super(SEMBA, self).train(mode)


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super(IdentityMessage, self).__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        
        mask = index < dim_size
        t = t[mask]
        msg = msg[mask]
        index = index[mask]
        
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):      
        mask = index < dim_size
        msg = msg[mask]
        index = index[mask]
        return scatter_mean(msg, index, dim=0, dim_size=dim_size)


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        t = t.to(torch.float)
        return self.lin(t.view(-1, 1)).cos()


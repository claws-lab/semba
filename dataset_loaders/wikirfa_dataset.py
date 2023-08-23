import os
import torch
import os.path as osp

from typing import Optional, Callable

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_gz, TemporalData
from datetime import datetime

class tgn_wikirfa(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 name = 'wikirfa',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        if self.name == 'wikirfa':
            self.url = 'https://snap.stanford.edu/data/wiki-RfA.txt.gz'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        if self.name == 'wikirfa':
            return 'wiki-RfA.txt'
        
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        print(path)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')
            data = [data[i:i + 7] for i in range(0, len(data), 8)][:-1]            
            data = [line for line in data if line[5][4:] != '']   #Removing dates with no entries
            data = [line for line in data if line[2][4:] != '0']  #Removing null votes
            
            #Sample entry
            #['SRC:Steel1943', 'TGT:BDD', 'VOT:1', 'RES:1', 'YEA:2013', 'DAT:23:13, 19 April 2013', "TXT:abc"]
            temp_data = []
            for line in data:
                try:
                    line[0] = line[0][4:]   #Removing 'SRC' from 'SRC:Steel1943'
                    line[1] = line[1][4:]   #Removing 'DST' from 'DST:BDD'
                    line[2] = line[2][4:]   #Removing 'VOT' from 'VOT:1'
                    
                    line[5] = (datetime.strptime(line[5][4:], "%H:%M, %d %B %Y") - datetime(1970, 1, 1)).total_seconds() #Converting time to total seconds from origin
                    
                    temp_data.append(line)
                except:
                    pass
            
            data = temp_data

            signs = [int(line[2]) for line in data]              
            edge_index = [[line[0], line[1]] for line in data]
            node_names = set()
            
            #Mapping node names to integers
            for src, dst in edge_index:
                node_names.add(src)
                node_names.add(dst)

            nodes = list(range(len(node_names)))
            mapping = {}
            for node, name in zip(nodes, node_names):
                mapping[name] = node

            for edge_id in range(len(edge_index)):
                edge_index[edge_id][0] = mapping[edge_index[edge_id][0]]
                edge_index[edge_id][1] = mapping[edge_index[edge_id][1]]

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_index = edge_index - edge_index.min()
            
#           edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            signs = torch.tensor(signs, dtype=torch.long)
            signs = signs > 0
            signs = signs * 1

            stamps_raw = [int(line[5]) for line in data]
            t = torch.tensor(stamps_raw).to(torch.long)
            t_sorted, ix = t.sort(descending=True) #Sort by descending, since the earliest date has most seconds passed

            edge_index = edge_index[:, ix]
            signs = signs[ix]
            
            src = edge_index[0]
            dst = edge_index[1]
            dst += int(src.max()) + 1
            msg = torch.ones(src.size(0), 1) #Set to 1, maybe changed to some edge weight
            t = t_sorted
            y = signs
            
            assert sorted(t.cpu().tolist(), reverse=True) == t.cpu().tolist()
            
#             print(edge_index.size())
#             print(signs.size())
#             print(t.size())
#             print(msg.size())

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
#         data.mapping = mapping

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

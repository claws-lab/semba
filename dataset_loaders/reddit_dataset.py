import os
import torch
import os.path as osp

from typing import Optional, Callable

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_gz, TemporalData
from datetime import datetime

class tgn_reddit(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 name = 'reddit',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        if self.name == 'reddit':
            self.url = 'https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        if self.name == 'reddit':
            return 'soc-redditHyperlinks-body.tsv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        print(path)
#         extract_gz(path, self.raw_dir)
#         os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[x for x in line.split('\t')] for line in data][1:]
            stamps_raw = [(datetime.strptime(line[3], "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds() for line in data]
            signs = [int(line[4]) for line in data]
            edge_attr = []
            for line in data:
                edge_attr.append(list(map(float, line[5:][0].split(','))))
                
            edge_index = [[line[0], line[1]] for line in data]
            node_names = set()
            for src, dst in edge_index:
                node_names.add(src)
                node_names.add(dst)


            nodes = list(range(len(node_names)))
            mapping = {}
            for node, name in zip(nodes, node_names):
                mapping[name] = node

        #     for side in [0, 1]:
            for edge_id in range(len(edge_index)):
                edge_index[edge_id][0] = mapping[edge_index[edge_id][0]]
                edge_index[edge_id][1] = mapping[edge_index[edge_id][1]]

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_index = edge_index - edge_index.min()

        #     edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            signs = torch.tensor(signs, dtype=torch.long)
            signs = signs > 0
            signs = signs * 1

            src = edge_index[0]
            dst = edge_index[1]
            dst += int(src.max()) + 1
            t = torch.tensor(stamps_raw).to(torch.long)
#             msg = torch.ones(src.size(0), 1)            
            msg = torch.tensor(edge_attr).to(torch.float)
            y = signs

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
#         data.mapping = mapping

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

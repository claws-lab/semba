import os
import torch
import os.path as osp

from typing import Optional, Callable

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_gz, TemporalData

class tgn_bitcoin(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 name = 'BitcoinOTC-1',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        if self.name == 'BitcoinOTC-1':
            self.url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
            
        elif self.name == 'BitcoinAlpha-1':
            self.url = 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz'
            
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        if self.name == 'BitcoinOTC-1':
            return 'soc-sign-bitcoinotc.csv'
        
        elif self.name == 'BitcoinAlpha-1':
            return 'soc-sign-bitcoinalpha.csv'
    

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[x for x in line.split(',')] for line in data]

            edge_index = [[int(line[0]), int(line[1])] for line in data]
            edge_attr = [float(line[2]) for line in data]
            stamps_raw = [int(float(line[3])) for line in data]

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_index = edge_index - edge_index.min()  # Start node IDs from 0 if it isn't

            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            signs = edge_attr > 0   # Negative edges become 0 and positive edges are 1
            signs = signs * 1

            t = torch.tensor(stamps_raw, dtype=torch.long)
            t_sorted, ix = t.sort()  # Sort the data accorrding to the timesteps. 

            edge_index_sorted = edge_index[:, ix]  # Reorder edges and signs according to the sorted indices
            signs_sorted = signs[ix]
            edge_attr_sorted = edge_attr[ix]

            src = edge_index_sorted[0]
            dst = edge_index_sorted[1]
            dst += int(src.max()) + 1   # To make graph bipartite, in case of bipaartite graphs
            t = t_sorted
            msg = torch.abs(edge_attr_sorted).unsqueeze(-1)  # Make absolute edge weights as edge features
            y = signs_sorted
            
            assert t_sorted.tolist() == sorted(stamps_raw)  # Ensure data is sorted sequentially
                        
        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

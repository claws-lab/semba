import os
import torch
import os.path as osp

from typing import Optional, Callable

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_tar, TemporalData
from datetime import datetime

class tgn_epinions(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 name = 'epinions',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        if self.name == 'epinions':
            self.url = 'http://konect.cc/files/download.tsv.epinions.tar.bz2'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        if self.name == 'epinions':
            return 'epinions/out.epinions'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode = 'r:bz2')
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[2:]
            data = [[x for x in line.split(' ')] for line in data][1:]
            
            edge_index, signs_raw, stamps_raw = [], [], []
            for line in data:
                if '' not in line:
                    edge_index.append([int(line[0]), int(line[1])])
                    signs_raw.append(float(line[2]))
                    stamps_raw.append(int(line[3]))

            signs_raw = torch.tensor(signs_raw, dtype=torch.float)
            mask_zero = (signs_raw != 0)
            signs = signs_raw.to(torch.long)[mask_zero]
            signs = (signs_raw > 0) * 1

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()[:, mask_zero]
            edge_index = edge_index - edge_index.min()
            t = torch.tensor(stamps_raw, dtype=torch.long)[mask_zero]
            t_sorted, ix = t.sort()

            edge_index = edge_index[:, ix]
            signs = signs[ix]

            src = edge_index[0]
            dst = edge_index[1]
            dst += int(src.max()) + 1
            msg = torch.ones(src.size(0), 1)
            y = signs

        data = TemporalData(src=src, dst=dst, t=t_sorted, msg=msg, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        
        
class tgn_wikiconflict_mini():
    
    def _init__(self):
        pass
    
    def get_data(self, cutoff=100000):
        with open('data/wikiconflict/raw/wikiconflict/out.wikiconflict', 'r') as f:
            data = f.read().split('\n')[2:cutoff]
            data = [[x for x in line.split(' ')] for line in data]
            edge_index, signs_raw, stamps_raw = [], [], []
            for line in data:
                if '' not in line:
                    edge_index.append([int(line[0]), int(line[1])])
                    signs_raw.append(float(line[2]))
                    stamps_raw.append(int(line[3]))

            signs_raw = torch.tensor(signs_raw, dtype=torch.float)
            mask_zero = (signs_raw != 0)
            signs = signs_raw.to(torch.long)[mask_zero]
            signs = (signs_raw > 0) * 1

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()[:, mask_zero]
            edge_index = edge_index - edge_index.min()
            t = torch.tensor(stamps_raw, dtype=torch.long)[mask_zero]
            t_sorted, ix = t.sort()

            edge_index = edge_index[:, ix]
            signs = signs[ix]

            src = edge_index[0]
            dst = edge_index[1]
            dst += int(src.max()) + 1
            msg = torch.ones(src.size(0), 1)
            y = signs
            
        dataset = [TemporalData(src=src, dst=dst, t=t_sorted, msg=msg, y=y)]
    
        return dataset

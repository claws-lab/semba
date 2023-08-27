import argparse

def parse_args (args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='semba')
    parser.add_argument('--dataset', type=str, default='BitcoinOTC-1')
    parser.add_argument('--task', type=str, default='sign_class')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--to_save', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr_init', default=0.01, type=float)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_feats', default=8, type=int)
    parser.add_argument('--feat_type', default="zeros", type=str)
    parser.add_argument('--early_stop_offset', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--neg_wt', default=1, type=float)
    parser.add_argument('--null_wt', default=1, type=float)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--null_nsamples', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args(args)
    
    return args
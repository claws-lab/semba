import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import pandas as pd

from dataset_loaders import *

__all__ = ['EarlyStopping', 'LRScheduler', 'get_embedding_loss', 'get_class_weights', 'get_dataset_stats', 'split_test_set_results', 'get_data'] 
    
class EarlyStopping():
    
    # Adapted from:
    # https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                

class LRScheduler():
    # Adapted from:
    # https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5, kind='plateau', verbose=True, mode='max'):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.kind = kind
        self.mode = mode
        if kind == 'plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                    self.optimizer,
                    mode=self.mode,
                    patience=self.patience,
                    factor=self.factor,
                    min_lr=self.min_lr,
                    verbose=verbose
                )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, 
                    step_size=50, 
                    gamma=factor,
                    verbose=verbose
                )
        
    def __call__(self, val_loss):
        if self.kind == 'plateau':
            self.lr_scheduler.step(val_loss)
        else:
            self.lr_scheduler.step()
        
        
def get_embedding_loss(z, src_pos, dst_pos, src_neg, dst_neg, null_dst_pos, null_dst_neg, assoc):
    
    out_1 = (z[assoc[src_pos]] - z[assoc[dst_pos]]).pow(2).sum(dim=1) - (z[assoc[src_pos]] - z[assoc[null_dst_pos]]).pow(2).sum(dim=1)
    loss_1 = torch.clamp(out_1, min=0).mean()
    
    out_2 = (z[assoc[src_neg]] - z[assoc[null_dst_neg]]).pow(2).sum(dim=1) - (z[assoc[src_neg]] - z[assoc[dst_neg]]).pow(2).sum(dim=1)
    loss_2 = torch.clamp(out_2, min=0).mean()
    
    return loss_1 + loss_2

def get_class_weights(labels, num_classes=3):
    
    if num_classes == 3:
        total_labels = torch.cat((labels, torch.tensor([2]*labels.size(0), device=labels.device)))
        w = compute_class_weight(class_weight='balanced', classes=[0, 1, 2], y=total_labels.cpu().numpy())
        
    elif num_classes == 2:
        w = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels.cpu().numpy())
        
    return torch.tensor(w, dtype=torch.float, device=labels.device)

def get_dataset_stats(train_data, val_data, test_data):
    
    train_pos_frac = sum(train_data.y) / train_data.y.size(0)
    val_data_frac = sum(val_data.y) / val_data.y.size(0)
    test_data_frac = sum(test_data.y) / test_data.y.size(0)
    
    print(f'Fraction of positive edges in train data - {train_pos_frac}')
    print(f'Fraction of positive edges in valid data - {val_data_frac}')
    print(f'Fraction of positive edges in test  data - {test_data_frac}')
    print()
    
    all_train_nodes = set(train_data.src.cpu().tolist()) | set(train_data.dst.cpu().tolist())
    all_val_nodes = set(val_data.src.cpu().tolist()) | set(val_data.dst.cpu().tolist())
    all_test_nodes = set(test_data.src.cpu().tolist()) | set(test_data.dst.cpu().tolist())

    print(f'Number of nodes in train data - {len(all_train_nodes)}')
    print(f'Number of nodes in val data - {len(all_val_nodes)}')
    print(f'Number of nodes in test data - {len(all_test_nodes)}')
    print()

    new_val_nodes = all_val_nodes - all_train_nodes
    new_test_nodes = all_test_nodes - all_train_nodes
    print(f'Number of new nodes in val data - {len(new_val_nodes)}')
    print(f'Number of new nodes in test data - {len(new_test_nodes)}')
    print()

    print(f'Fraction of new nodes in val data - {len(new_val_nodes)/ len(all_val_nodes)}')
    print(f'Fraction of new nodes in test data - {len(new_test_nodes) / len(all_test_nodes)}')
    print()
    
    for name, inference_data in [('Val data', val_data), ('Test data', test_data)]:

        both_new_node_edges, one_new_node_edges, no_new_node_edges = [], [], []
        both_new_node_edges_sign, one_new_node_edges_sign, no_new_node_edges_sign = [], [], []
        for s, d, sign in zip(inference_data.src.tolist(), inference_data.dst.tolist(), inference_data.y.tolist()):

            if (s in all_train_nodes) and (d in all_train_nodes):
                no_new_node_edges.append(1)
                no_new_node_edges_sign.append(sign)
            else:
                no_new_node_edges.append(0)   

            if (s not in all_train_nodes) ^ (d not in all_train_nodes):
                one_new_node_edges.append(1)
                one_new_node_edges_sign.append(sign)
            else:
                one_new_node_edges.append(0)

            if (s not in all_train_nodes) and (d not in all_train_nodes):
                both_new_node_edges.append(1)
                both_new_node_edges_sign.append(sign)
            else:
                both_new_node_edges.append(0) 

        no_new_node_frac = sum(no_new_node_edges) / inference_data.num_events
        one_new_node_frac = sum(one_new_node_edges) / inference_data.num_events
        two_new_node_frac = sum(both_new_node_edges) / inference_data.num_events       
        
        no_new_node_frac_pos = sum(no_new_node_edges_sign) / len(no_new_node_edges_sign)
        one_new_node_frac_pos = sum(one_new_node_edges_sign) / len(one_new_node_edges_sign)
        two_new_node_frac_pos = sum(both_new_node_edges_sign) / len(both_new_node_edges_sign)
        
        print(f'Frac. of edges with 0 new nodes in {name} - {no_new_node_frac:.4f} - (+ve - {no_new_node_frac_pos:.4f}, -ve - {1-no_new_node_frac_pos:.4f})')
        print(f'Frac. of edges with 1 new nodes in {name} - {one_new_node_frac:.4f} - (+ve - {one_new_node_frac_pos:.4f}, -ve - {1-one_new_node_frac_pos:.4f})')
        print(f'Frac. of edges with 2 new nodes in {name} - {two_new_node_frac:.4f} - (+ve - {two_new_node_frac_pos:.4f}, -ve - {1-two_new_node_frac_pos:.4f})')
        print()
        
        no_new_node_edges = torch.tensor(no_new_node_edges, device=train_data.src.device, dtype=torch.long)
        one_new_node_edges = torch.tensor(one_new_node_edges, device=train_data.src.device, dtype=torch.long)
        both_new_node_edges = torch.tensor(both_new_node_edges, device=train_data.src.device, dtype=torch.long)

        no_new_node_edges = no_new_node_edges == 1
        one_new_node_edges = one_new_node_edges == 1 
        both_new_node_edges = both_new_node_edges == 1 
        
    return no_new_node_edges, one_new_node_edges, both_new_node_edges
    
def split_test_set_results(inference_data, probabilities, pred, true, new_node_masks):
    
    no_new_node_edges, one_new_node_edges, both_new_node_edges = new_node_masks
    
    prob = torch.tensor(probabilities, device=inference_data.src.device)
    pred = torch.tensor(pred, device=inference_data.src.device)
    true = torch.tensor(true, device=inference_data.src.device)

    prob_exi = prob[no_new_node_edges]
    pred_exi = pred[no_new_node_edges]
    true_exi = true[no_new_node_edges]

    prob_one = prob[one_new_node_edges]
    pred_one = pred[one_new_node_edges]
    true_one = true[one_new_node_edges]

    prob_two = prob[both_new_node_edges]
    pred_two = pred[both_new_node_edges]
    true_two = true[both_new_node_edges]

    cm_exi = confusion_matrix(true_exi.cpu(), pred_exi.cpu())
    roc_exi = roc_auc_score(true_exi.cpu(), prob_exi.cpu(), average='weighted')
    fpr1_exi = cm_exi[0][1] / (cm_exi[0][1] + cm_exi[0][0])
    f1_exi = f1_score(true_exi.cpu(), pred_exi.cpu(), average='weighted')
    
    cm_one = confusion_matrix(true_one.cpu(), pred_one.cpu())
    roc_one = roc_auc_score(true_one.cpu(), prob_one.cpu(), average='weighted')
    fpr1_one = cm_one[0][1] / (cm_one[0][1] + cm_one[0][0])
    f1_one = f1_score(true_one.cpu(), pred_one.cpu(), average='weighted')
    
    cm_two = confusion_matrix(true_two.cpu(), pred_two.cpu())
    roc_two = roc_auc_score(true_two.cpu(), prob_two.cpu(), average='weighted')
    fpr1_two = cm_two[0][1] / (cm_two[0][1] + cm_two[0][0])
    f1_two = f1_score(true_two.cpu(), pred_two.cpu(), average='weighted')
      
    print()
    print(f'Results for no new nodes - ROC_AUC: {roc_exi:.4f}, F1: {fpr1_exi:.4f}, FPR: {f1_exi:.4f}')
    print(f'Results for one new nodes - ROC_AUC: {roc_one:.4f}, F1: {fpr1_one:.4f}, FPR: {f1_one:.4f}')
    print(f'Results for two new nodes - ROC_AUC: {roc_two:.4f}, F1: {fpr1_two:.4f}, FPR: {f1_two:.4f}')
    print()
    
def get_data(NAME, path, device, val_ratio=0.15, test_ratio=0.15):

    if NAME == 'BitcoinOTC-1' or NAME == 'BitcoinAlpha-1':
        dataset = tgn_bitcoin(path, edge_window_size=1, name=NAME)
    elif NAME == 'epinions':
        dataset = tgn_epinions(path, edge_window_size=1, name=NAME)
    elif NAME == 'wikirfa':
        dataset = tgn_wikirfa(path, edge_window_size=1, name=NAME)

    data = dataset[0].to(device)

    train_data, val_data, test_data = data.train_val_test_split(val_ratio=val_ratio, test_ratio=test_ratio)

    return data, train_data, val_data, test_data
    


def seq_batches (data, batch_size=128):
    curr_idx = 0
    while curr_idx < len(data):
        yield data[curr_idx:curr_idx+batch_size]
        curr_idx += batch_size
    
    

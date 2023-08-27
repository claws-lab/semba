import os
from csv import DictWriter
from collections import defaultdict
import numpy as np
import random
import os.path as osp
import time
import sys
import pickle as pkl

import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
# from torch_geometric.nn import SignedGCN
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from parser import parse_args
from utils import *
from model_wrapper import STGNN


torch.autograd.set_detect_anomaly(True)

def train(args):
    model.train()
    total_loss, probabilities, pred, true = 0, [], [], []
    total_events = 0
    for batch_id, batch in enumerate(train_data.seq_batches(batch_size=args.batch_size)):
        optimizer.zero_grad()
        
        src, dst, t, weight, signs = batch.src, batch.dst, batch.t, batch.msg, batch.y
        src_pos, dst_pos, t_pos, weight_pos = src[signs == 1], dst[signs == 1], \
                                                t[signs == 1], weight[signs == 1]
        src_neg, dst_neg, t_neg, weight_neg = src[signs == 0], dst[signs == 0], \
                                                t[signs == 0], weight[signs == 0]

        pos_ei_batch = torch.stack([src_pos, dst_pos])
        neg_ei_batch = torch.stack([src_neg, dst_neg])
        ei_batch = torch.cat([pos_ei_batch, neg_ei_batch], dim=1)
        null_ei_batch = None
        signed_edge_weights = None

        if args.task == "link_pred":
            null_ei_batch = negative_sampling(ei_batch, num_nodes=x.size(0), 
                                    num_neg_samples=args.null_nsamples*ei_batch.shape[1]).to(device=args.device)
            y = torch.cat([torch.ones_like(src), torch.zeros_like(null_ei_batch[0])]).long()
            src = torch.cat([src, null_ei_batch[0]])
            dst = torch.cat([dst, null_ei_batch[1]])
        elif args.task == "sign_class":
            y = signs
        elif args.task == "signlink_class":
            null_ei_batch = negative_sampling(ei_batch, num_nodes=x.size(0), 
                                    num_neg_samples=args.null_nsamples*ei_batch.shape[1]).to(device=args.device)
            src = torch.cat([src_pos, src_neg, null_ei_batch[0]])
            dst = torch.cat([dst_pos, dst_neg, null_ei_batch[1]])
            y = torch.cat([torch.zeros_like(src_pos), torch.ones_like(src_neg), 
                            2*torch.ones_like(null_ei_batch[0])]).long()
        elif args.task == "signwt_pred":
            signed_edge_weights = signs * weight.float().mean(dim=1)
            y = signed_edge_weights
        
        z = model (x, pos_ei_batch, neg_ei_batch, t_pos, t_neg, weight_pos, weight_neg, 
                   to_update=True).to(device=args.device)
        loss = model.loss (z, pos_ei_batch, neg_ei_batch, null_edge_index=null_ei_batch, 
                           signed_edge_weights=signed_edge_weights, neg_wt=args.neg_wt, null_wt=args.null_wt)
        prob = model.predict (z, src, dst).detach()
        y_pred = model.predict (z, src, dst, classify=True).detach()
        total_loss += float(loss) * len(src)
        total_events += len(src)
        
        loss.backward()
        optimizer.step()

        probabilities.extend(prob.tolist())
        pred.extend(y_pred.tolist())
        true.extend(y.tolist())
        
    if args.task == 'signlink_class':
        f1_wt = f1_score(true, pred, average='weighted')
        f1_macro = f1_score(true, pred, average='macro')
        f1_micro = f1_score(true, pred, average='micro')
        auc_wt = roc_auc_score(true, probabilities, multi_class='ovr', average='weighted')
        auc_macro = roc_auc_score(true, probabilities, multi_class='ovr', average='macro')
        return total_loss / total_events, [f1_wt, f1_macro, f1_micro, auc_wt, auc_macro]
    elif args.task == 'signwt_pred':
        rmse = np.sqrt(mean_squared_error(true, pred))
        pcc, _ = pearsonr(true, pred)
        r2 = r2_score(true, pred)
        return total_loss / total_events, [rmse, pcc, r2]
    else:
        auc = roc_auc_score(true, probabilities)
        f1_wt = f1_score(true, pred, average='weighted')
        f1_bin = f1_score(true, pred)
        return total_loss / total_events, [auc, f1_wt, f1_bin]


@torch.no_grad()
def test(args, inference_data, inference_type='val'):
    model.eval()
    total_loss, probabilities, pred, true = 0, [], [], []
    total_events = 0
    
    if inference_type == 'test':
        pos_edge_index_batch = torch.cat([pos_edge_index_train, pos_edge_index_val], -1)
        neg_edge_index_batch = torch.cat([neg_edge_index_train, neg_edge_index_val], -1)
        
    for batch_id, batch in enumerate(inference_data.seq_batches(batch_size=args.batch_size)):

        src, dst, t, weight, signs = batch.src, batch.dst, batch.t, batch.msg, batch.y
        src_pos, dst_pos, t_pos, weight_pos = src[signs == 1], dst[signs == 1], \
                                                t[signs == 1], weight[signs == 1]
        src_neg, dst_neg, t_neg, weight_neg = src[signs == 0], dst[signs == 0], \
                                                t[signs == 0], weight[signs == 0]

        pos_ei_batch = torch.stack([src_pos, dst_pos])
        neg_ei_batch = torch.stack([src_neg, dst_neg])
        ei_batch = torch.cat([pos_ei_batch, neg_ei_batch], dim=1)
        null_ei_batch = None
        signed_edge_weights = None

        if args.task == "link_pred":
            null_ei_batch = negative_sampling(ei_batch, num_nodes=x.size(0), 
                                    num_neg_samples=1*ei_batch.shape[1]).to(device=args.device)
            y = torch.cat([torch.ones_like(src), torch.zeros_like(null_ei_batch[0])])
            src = torch.cat([src, null_ei_batch[0]])
            dst = torch.cat([dst, null_ei_batch[1]])
        elif args.task == "sign_class":
            y = signs
        elif args.task == "signlink_class":
            null_ei_batch = negative_sampling(ei_batch, num_nodes=x.size(0), 
                                    num_neg_samples=1*ei_batch.shape[1]).to(device=args.device)
            src = torch.cat([src_pos, src_neg, null_ei_batch[0]])
            dst = torch.cat([dst_pos, dst_neg, null_ei_batch[1]])
            y = torch.cat([torch.zeros_like(src_pos), torch.ones_like(src_neg), 2*torch.ones_like(null_ei_batch[0])])
        elif args.task == "signwt_pred":
            signed_edge_weights = signs * weight.float().mean(dim=1)
            y = signed_edge_weights
        
        z = model (x, pos_ei_batch, neg_ei_batch, t_pos, t_neg, weight_pos, weight_neg, 
                   to_update=True).to(device=args.device)
        loss = model.loss (z, pos_ei_batch, neg_ei_batch, null_edge_index=null_ei_batch, 
                           signed_edge_weights=signed_edge_weights, neg_wt=args.neg_wt, null_wt=args.null_wt)
        prob = model.predict (z, src, dst).detach()
        y_pred = model.predict (z, src, dst, classify=True).detach()
        total_loss += float(loss) * len(src)
        total_events += len(src)

        probabilities.extend(prob.tolist())
        pred.extend(y_pred.tolist())
        true.extend(y.tolist())
        
    if args.task == 'signlink_class':
        f1_wt = f1_score(true, pred, average='weighted')
        f1_macro = f1_score(true, pred, average='macro')
        f1_micro = f1_score(true, pred, average='micro')
        auc_wt = roc_auc_score(true, probabilities, multi_class='ovr', average='weighted')
        auc_macro = roc_auc_score(true, probabilities, multi_class='ovr', average='macro')
        return total_loss / total_events, [f1_wt, f1_macro, f1_micro, auc_wt, auc_macro]
    elif args.task == 'signwt_pred':
        rmse = np.sqrt(mean_squared_error(true, pred))
        pcc, _ = pearsonr(true, pred)
        r2 = r2_score(true, pred)
        return total_loss / total_events, [rmse, pcc, r2]
    else:
        auc = roc_auc_score(true, probabilities)
        f1_wt = f1_score(true, pred, average='weighted')
        f1_bin = f1_score(true, pred)
        return total_loss / total_events, [auc, f1_wt, f1_bin]


if __name__ == '__main__':
    args = parse_args (sys.argv[1:])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_path = osp.join('./data', args.dataset)
    data, train_data, val_data, test_data = get_data(args.dataset, dataset_path, args.device, val_ratio=args.val_ratio, 
                                                    test_ratio=args.test_ratio)
    # transductive
    train_nodes = torch.stack([train_data.src, train_data.dst]).unique()
    train_nodes_yes = torch.zeros(data.num_nodes, dtype=bool, device=args.device)
    train_nodes_yes[train_nodes] = True
    val_idx = train_nodes_yes[val_data.src] & train_nodes_yes[val_data.dst]
    test_idx = train_nodes_yes[test_data.src] & train_nodes_yes[test_data.dst]
    val_trans_data, test_trans_data = val_data[val_idx], test_data[test_idx]
    # inductive
    train_nodes = torch.stack([train_data.src, train_data.dst]).unique()
    train_nodes_no = torch.ones(data.num_nodes, dtype=bool, device=args.device)
    train_nodes_no[train_nodes] = False
    val_idx = train_nodes_no[val_data.src] & train_nodes_no[val_data.dst]
    test_idx = train_nodes_no[test_data.src] & train_nodes_no[test_data.dst]
    val_ind_data, test_ind_data = val_data[val_idx], test_data[test_idx]
    
    
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    weights = get_class_weights(data.y, num_classes=2)
    # print (weights)
    # neg_wt = weights[0]/weights[1]

    edge_index_train = torch.stack([train_data.src, train_data.dst], dim=0)
    edge_index_val = torch.stack([val_data.src, val_data.dst], dim=0) 

    pos_edge_index_train = edge_index_train[:, train_data.y == 1]
    neg_edge_index_train = edge_index_train[:, train_data.y == 0]

    pos_edge_index_val = edge_index_val[:, val_data.y == 1]
    neg_edge_index_val = edge_index_val[:, val_data.y == 0]

    num_feats = args.num_feats
    if args.feat_type == 'one-hot':
        num_feats = data.num_nodes
        x = torch.diag(data.num_nodes, dtype=torch.float, device=args.device)
    elif args.feat_type == 'random':
        x = torch.rand(data.num_nodes, num_feats, dtype=torch.float, device=args.device)
    elif args.feat_type == 'zeros':
        x = torch.zeros(data.num_nodes, num_feats, dtype=torch.float, device=args.device)
    # x = model.create_spectral_features(pos_edge_index, neg_edge_index, data.num_nodes)
    # elif args.feat_type == 'degree':
    #     x = torch.zeros((data.num_nodes, num_feats), dtype=torch.float, device=args.device)
    #     train_data.src
    # x = torch.arange(data.num_nodes+1, dtype=torch.float).unsqueeze(-1).to(device=args.device)

    train_cm, test_cm = [], []
    train_losses, val_losses = [], []
    total_time = 0

    model = STGNN (args.model, args.task, num_feats, data.num_nodes, args.embedding_dim, args.num_layers, device=args.device, debug=args.debug)

    optimizer = torch.optim.Adam(set(model.parameters()), lr=args.lr_init)
    # early_stopping = EarlyStopping(patience=5)
    lr_scheduler = LRScheduler(optimizer)
    if args.task == 'signlink_class':
        metrics = ["F1_wt", "F1_macro", "F1_micro", "AUC_wt", "AUC_macro"]
    elif args.task == 'signwt_pred':
        metrics = ["RMSE", "PCC", "R2", "KL"]
    else:
        metrics = ["AUC", "F1_wt", "F1_bin"]
    metric_string = lambda x: ' '.join([f'{metric}: {param:.4f}' for metric, param in zip(metrics, x)])
    for epoch in range(1, args.num_epochs + 1):
        model.reset_memory()
        start = time.time()
        train_loss, train_params = train(args)
        time_epch = time.time() - start
        total_time += time_epch
        val_loss, val_params = test(args, val_data, epoch)
        lr_scheduler(val_loss)
        print(f'E{epoch:002d} Tr [Loss: {train_loss:.4f} {metric_string(train_params)}]')
        print(f'Val [Loss: {val_loss:.4f} {metric_string(train_params)}]')
        print(f'Time: {time_epch}')
        print()
    
    test_loss, test_params = test(args, test_data, epoch, 'test')
    test_trans_loss, test_trans_params = test(args, test_trans_data, epoch, 'test')
    test_ind_loss, test_ind_params = test(args, test_ind_data, epoch, 'test')
    print(f"Time taken: {total_time}")
    print('*****************')
    print('Overall performance')
    print('*****************')
    print(f'Test [Loss: {test_loss:.4f} {metric_string(test_params)}]')
    print(f'Test [Loss: {test_trans_loss:.4f} {metric_string(test_trans_params)}]')
    print(f'Test [Loss: {test_ind_loss:.4f} {metric_string(test_ind_params)}]')

    if args.to_save:
        if args.model == 'semba':
            PATH = f'src/saved_models/{args.dataset}/{args.task}/'
        else:
            PATH = f'baselines/{args.model}/saved_models/{args.dataset}/{args.task}/'
        try:
            os.makedirs(PATH)
        except:
            pass
        pkl.dump ({'model_name': args.model, 'task': args.task, 'num_feats': num_feats, 'num_nodes': data.num_nodes, 
                   'embedding_dim': args.embedding_dim, 'num_layers': args.num_layers, 'device': args.device},
                  open (f'{PATH}/model_params.pkl', 'wb'))
        torch.save (model.state_dict(), f'{PATH}/model_state.pt')
        metrics_dict = {metric:param for metric, param in zip(metrics, test_params)}
        metrics_fields = ['Model', 'Dataset', 'Nepochs', 'LRin'] + list(metrics_dict.keys())
        metrics_dict['Model'] = args.model
        metrics_dict['Dataset'] = args.dataset
        metrics_dict['Nepochs'] = args.num_epochs
        metrics_dict['LRin'] = args.lr_init
        if (not (os.path.exists(f"test_summ_{args.task}.csv"))):
            DictWriter(open (f"test_summ_{args.task}.csv", "w+"), fieldnames=metrics_fields).writeheader()
            DictWriter(open (f"testTrans_summ_{args.task}.csv", "w+"), fieldnames=metrics_fields).writeheader()
            DictWriter(open (f"testInd_summ_{args.task}.csv", "w+"), fieldnames=metrics_fields).writeheader()
        DictWriter(open (f"test_summ_{args.task}.csv", "a"), fieldnames=metrics_fields).writerow(metrics_dict)
        metrics_dict = {metric:param for metric, param in zip(metrics, test_trans_params)}
        metrics_dict['Model'] = args.model
        metrics_dict['Dataset'] = args.dataset
        metrics_dict['Nepochs'] = args.num_epochs
        metrics_dict['LRin'] = args.lr_init
        DictWriter(open (f"testTrans_summ_{args.task}.csv", "a"), fieldnames=metrics_fields).writerow(metrics_dict)
        metrics_dict = {metric:param for metric, param in zip(metrics, test_ind_params)}
        metrics_dict['Model'] = args.model
        metrics_dict['Dataset'] = args.dataset
        metrics_dict['Nepochs'] = args.num_epochs
        metrics_dict['LRin'] = args.lr_init
        DictWriter(open (f"testInd_summ_{args.task}.csv", "a"), fieldnames=metrics_fields).writerow(metrics_dict)

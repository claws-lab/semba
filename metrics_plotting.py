import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import entropy
import matplotlib.pyplot as plt
from torch.nn import functional as F


__all__ = ['MetricsLog', 'plot_metrics_from_history', 'get_kl_divergence']

class MetricsLog:
    
    def __init__(self, list_of_metrics):
        
        self.metrics = {}
        self.list_of_metrics = list_of_metrics
        self.num_epochs_logged = 0
        
        for metric in list_of_metrics:
            self.metrics[metric] = {'train': [], 'val': [], 'test': []}
            
    def return_metrics(self):
        self.num_epochs_logged = len(self.metrics[self.list_of_metrics[0]]['train'])
        return self.metrics
    
    def log_this_epoch(self, split, values):
        
        for metric, metric_value in zip(self.list_of_metrics, values):
            self.metrics[metric][split].append(metric_value)
            

def plot_metrics_from_history(metrics_log, DATASET, MODEL, TASK, to_save=False):

    history = metrics_log.return_metrics()
    
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics_log.list_of_metrics):
        train_list, val_list, test_list = history[metric]['train'], history[metric]['val'], history[metric]['test']
        
        epochs = list(range(1, metrics_log.num_epochs_logged+1))
        
        plt.subplot(2, 2, i+1)
        plt.plot(epochs, train_list, label='train ' + metric)
        plt.plot(epochs, val_list, label='val ' + metric)
        plt.plot(epochs, test_list, label='test ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(f'{TASK} - {DATASET} - {MODEL} - {metric}')
        plt.legend()
        
    plt.tight_layout()
    if to_save:
        PATH = f'baselines/{MODEL}/saved_models/{DATASET}/{TASK}/'
        plt.savefig(f"{PATH}/model_train.png")
    
def get_rmse_pcc_negative_edges(true, pred):
    
    pred = torch.tensor(pred)
    true = torch.tensor(true)
    
    true_neg = true[true < 0]
    pred_neg = pred[true < 0]
    
    rmse = np.sqrt(F.mse_loss(true_neg, pred_neg))
    pcc, _ = pearsonr(true_neg, pred_neg)
    
    return rmse, pcc

def get_kl_divergence(true, pred):
    
    predictions = torch.round(torch.tensor(pred))
    edge_weights = torch.tensor(true).float()    
    
    predictions_series = pd.Series(predictions.tolist()).value_counts()
    weights_series = pd.Series(edge_weights.tolist()).value_counts()
    
    pred_counts, true_counts, total = [], [], len(pred)
    for i in range(-10, 11):
        weight = i*1.0
        pred_counts.append(predictions_series.get(weight, 1e-8) / total)
        true_counts.append(weights_series.get(weight, 1e-8) / total)
        
    kl = entropy(pk=true_counts, qk=pred_counts)
    
    return kl

def plot_weight_distribution(true, pred):

    predictions = torch.round(torch.tensor(pred))
    edge_weights = torch.tensor(true).float()

#     edge_weights_pos = edge_weights[edge_weights > 0]
#     edge_weights_neg = edge_weights[edge_weights < 0]

#     predictions_pos = predictions[predictions > 0]
#     predictions_neg = predictions[predictions < 0]

    predictions_series = pd.Series(predictions.tolist())
    weights_series = pd.Series(edge_weights.tolist())

#     predictions_series_pos = pd.Series(predictions_pos.tolist())
#     predictions_series_neg = pd.Series(predictions_neg.tolist())

#     weights_series_pos = pd.Series(edge_weights_pos.tolist())
#     weights_series_neg = pd.Series(edge_weights_neg.tolist())

    true_counts = weights_series.value_counts().sort_index()
    pred_counts = predictions_series.value_counts().sort_index()
    
    merged_counts = {}
    for i in range(-10, 11):
        if i == 0:
            continue
        weight = i*1.0
        merged_counts[weight] = [true_counts.to_dict().get(weight, 0),  pred_counts.to_dict().get(weight, 0)]
    
    merged_counts = pd.DataFrame.from_dict(merged_counts, orient='index', columns=['true weights', 'predicted weights'])
    
    merged_counts.plot(kind='bar', xlabel='edge weight', ylabel='counts', title='Distribution of true and predicted weights')
    
    
#     fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
#     ax1 = axs[0]
#     ax2 = axs[1]

#     pos_counts = weights_series_pos.value_counts().sort_index()
#     neg_counts = weights_series_neg.value_counts().sort_index(ascending=False)

#     pred_pos_counts = predictions_series_pos.value_counts().sort_index()
#     pred_neg_counts = predictions_series_neg.value_counts().sort_index(ascending=False)

#     merged_counts_pos, merged_counts_neg = {}, {}
#     for i in range(1, 11):
#         weight = i*1.0
#         merged_counts_pos[weight] = [pos_counts.to_dict().get(weight, 0),  pred_pos_counts.to_dict().get(weight, 0)]
#         merged_counts_neg[-weight] = [neg_counts.to_dict().get(-weight, 0),  pred_neg_counts.to_dict().get(-weight, 0)]

#     merged_counts_pos = pd.DataFrame.from_dict(merged_counts_pos, orient='index', columns=['true weights', 'predicted weights'])
#     merged_counts_neg = pd.DataFrame.from_dict(merged_counts_neg, orient='index', columns=['true weights', 'predicted weights'])

# #     pos_counts.plot(ax=ax1, kind='bar', xlabel='edge weight', ylabel='counts')
# #     neg_counts.plot(ax=ax2, kind='bar', xlabel='edge weight', ylabel='counts')
#     merged_counts_pos.plot(ax=ax1, kind='bar', xlabel='edge weight', ylabel='counts')
#     merged_counts_neg.plot(ax=ax2, kind='bar', xlabel='edge weight', ylabel='counts')
    
def plot_metrics(losses, rocs, f1s, fprs, name, MODEL):
    
    train_rocs, val_rocs, test_rocs = rocs
    train_losses, val_losses, test_losses = losses
    train_f1s, val_f1s, test_f1s = f1s
    train_fprs, val_fprs, test_fprs = fprs
    
    epochs = list(range(1, len(train_losses)+1))

    plt.figure(figsize=(14, 10)) 
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='train_loss')
    plt.plot(epochs, val_losses, label='val_loss')
    plt.plot(epochs, test_losses, label='test_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title(f'{name} - {MODEL} - Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_rocs, label='train_roc')
    plt.plot(epochs, val_rocs, label='val_roc')
    plt.plot(epochs, test_rocs, label='test_roc')
    plt.xlabel("Epochs")
    plt.ylabel("ROC-AUC")
    plt.title(f'{name} - {MODEL} - ROC-AUC')
    plt.legend()
        
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, label='train_f1s')
    plt.plot(epochs, val_f1s, label='val_f1s')
    plt.plot(epochs, test_f1s, label='test_f1s')
    plt.xlabel("Epochs")
    plt.ylabel("F1")
    plt.title(f'{name} - {MODEL} - F1')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_fprs, label='train_fprs')
    plt.plot(epochs, val_fprs, label='val_fprs')
    plt.plot(epochs, test_fprs, label='test_fprs')
    plt.xlabel("Epochs")
    plt.ylabel("FPR")
    plt.title(f'{name} - {MODEL} - FPR')
    plt.legend()
    
    plt.tight_layout()
    
    
def plot_metrics_weight_pred(rmses, pccs, r2s, NAME, MODEL):
    
    train_rmses, val_rmses, test_rmses = rmses
    train_pccs, val_pccs, test_pccs = pccs
    train_r2s, val_r2s, test_r2s = r2s
    
    epochs = list(range(1, len(rmses)+1))

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    epochs = list(range(1, len(train_rmses)+1))
    plt.plot(epochs, train_rmses, label='train_loss')
    plt.plot(epochs, val_rmses, label='val_loss')
    plt.plot(epochs, test_rmses, label='test_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title(f'{NAME} - {MODEL} - RMSE Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_pccs, label='train_pcc')
    plt.plot(epochs, val_pccs, label='val_pcc')
    plt.plot(epochs, test_pccs, label='test_pcc')
    plt.xlabel("Epochs")
    plt.ylabel("PCC")
    plt.title(f'{NAME} - {MODEL} - PCC')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_r2s, label='train_r2')
    plt.plot(epochs, val_r2s, label='val_r2')
    plt.plot(epochs, test_r2s, label='test_r2')
    plt.xlabel("Epochs")
    plt.ylabel("R2")
    plt.title(f'{NAME} - {MODEL} - R2')
    plt.legend()

    plt.tight_layout()
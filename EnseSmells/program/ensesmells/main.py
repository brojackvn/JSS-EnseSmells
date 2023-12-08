# lib pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# lib default python
import os
import random
import time
import gc

# lib science math
import numpy as np

# lib sklearn
from sklearn.model_selection import StratifiedKFold, KFold

# my class
import config
import utils
import data
import train
from model import CNN_LSTM, CNN_BiLSTM, CNN_LSTM_METRICS, calculate_size_lstm
import time
import datetime
from config import argument
# my tool
from utils import write_file

args = argument()

if __name__ == "__main__":
    pos_weight_set = [
        torch.tensor(1.0, dtype=torch.float), 
        torch.tensor(2.0, dtype=torch.float), 
        torch.tensor(4.0, dtype=torch.float),
        torch.tensor(8.0, dtype=torch.float), 
        torch.tensor(12.0, dtype=torch.float), 
        torch.tensor(32.0, dtype=torch.float),
        torch.tensor(84.0, dtype=torch.float)
    ]

    kernel_size_set = [3, 4, 5, 6, 7]
    now = datetime.datetime.now()

    data_path = args.data_path

    result_summary = {}

    for pos_weight in pos_weight_set:
        for kernel_size in kernel_size_set:
            smell, model = utils.get_smell_and_model(args.data_path)   
            
            file_name = f'{model}_{smell}_{now.strftime("%d%m%Y_%H%M")}_posweight_{pos_weight.item()}_kernel_{kernel_size}'
            track_file = f'{args.tracking_dir}/{file_name}.txt'
            result_file = f'{args.result_dir}/{model}_{smell}_{now.strftime("%d%m%Y_%H%M")}.txt'
            
            precision = []
            recall = []
            f1 = []
            auc = []
            mcc = []
            
            if args.model == 'EnseSmells':
                load_dataset = enumerate(utils.get_CombineData_pickle(data_path))
            elif args.model == 'DeepSmells_TokenIndexing':
                load_dataset = enumerate(utils.get_data_token_indexing(data_path))
            elif args.model == 'EnseSmell_TokenIndexing':
                load_dataset = enumerate(utils.get_data_token_indexing_COMBINING(data_path))
            else: # DeepSmells
                load_dataset = enumerate(utils.get_data_pickle(data_path))

            for index, datasets in load_dataset:
                print(f"{'=+'*25} FOLD: {index+1} / 5 {'+='*25}")
                write_file(track_file, f"{'=+'*25} FOLD: {index+1} / 5 {'+='*25}\n")
                # Sample elements randomly from given list of ids, no replacement
                if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                    train_set = data.DatasetCombine(datasets.train_data, datasets.train_data_metrics, datasets.train_labels)
                    valid_set = data.DatasetCombine(datasets.eval_data, datasets.eval_data_metrics, datasets.eval_labels)
                if args.model == 'DeepSmells' or args.model == 'DeepSmells_TokenIndexing': 
                    train_set = data.Dataset(datasets.train_data, datasets.train_labels)
                    valid_set = data.Dataset(datasets.eval_data, datasets.eval_labels)

                # Define data loaders for training and testing data in this fold  
                train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True)
                valid_loader = DataLoader(valid_set, batch_size=args.valid_batchsize, shuffle=True)

                length_code = train_set[0][0].size()[1]
                # Calculate size LSTM - CC
                input_size_lstm = calculate_size_lstm(input_size=length_code, kernel_size=kernel_size)

                # Initialize the model, optimizer, scheduler, loss
                if args.model == 'DeepSmells' or args.model == 'DeepSmells_TokenIndexing':
                    model = CNN_LSTM(kernel_size = kernel_size, input_size_lstm=input_size_lstm, hidden_size_lstm=args.hidden_size_lstm).to(config.Config.DEVICE)
                if args.model == 'DeepSmells-BiLSTM':
                    model = CNN_BiLSTM(kernel_size = kernel_size, input_size_lstm=input_size_lstm, hidden_size_lstm=args.hidden_size_lstm).to(config.Config.DEVICE)
                if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                    metrics_size = train_set[0][1].size()[0]
                    model = CNN_LSTM_METRICS(kernel_size = kernel_size, input_size_lstm=input_size_lstm, hidden_size_lstm=args.hidden_size_lstm, metrics_size = metrics_size).to(config.Config.DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
                # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

                train_loss_fn, valid_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight), nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                trainer = train.Trainer(
                    device = config.Config.DEVICE,
                    dataloader = (train_loader, valid_loader),
                    model = model,
                    loss_fns = (train_loss_fn, valid_loss_fn),
                    optimizer = optimizer,
                    # scheduler = step_lr_scheduler,
                )

                best_precision, best_recall, best_f1, best_auc, best_mcc = trainer.fit(
                    epochs = args.nb_epochs,
                    checkpoint_dir = None,
                    custom_name = file_name,
                    track_dir = track_file,
                    threshold = args.threshold,
                )

                precision.append(best_precision)
                recall.append(best_recall)
                f1.append(best_f1)
                auc.append(best_auc)
                mcc.append(best_mcc)
                # del model, optimizer, train_loss_fn, valid_loss_fn, trainer, best_precision, best_recall, best_f1, best_mcc
                gc.collect()
            
            result_summary[f"pos_weight_{pos_weight.item()}_kernel_{kernel_size}"] = [sum(precision)/len(precision), 
                                                                                  sum(recall)/len(recall), 
                                                                                  sum(f1)/len(f1), 
                                                                                  sum(auc)/len(auc), 
                                                                                  sum(mcc)/len(mcc)                                                                                                                                        
                                                                                ]
            write_file(result_file, f"pos_weight_{pos_weight.item()}_kernel_{kernel_size},{sum(precision)/len(precision)},{sum(recall)/len(recall)},{sum(f1)/len(f1)},{sum(auc)/len(auc)},{sum(mcc)/len(mcc)}\n")
    max_key = None
    max_f1 = 0
    for key, value in result_summary.items():
        # write_file(result_file, f"{key},{value[0]},{value[1]},{value[2]},{value[3]},{value[4]}\n")
        if value[2] > max_f1:
            max_f1 = value[2]
            max_key = key

    write_file(result_file, f"BEST-{max_key},{result_summary[max_key][0]}, {result_summary[max_key][1]}, {result_summary[max_key][2]}, {result_summary[max_key][3]}, {result_summary[max_key][4]}\n")

import torch
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
import utils
import argparse

def argument():
    parser = argparse.ArgumentParser(description='Hyperparameter')

    parser.add_argument('--data_path', type=str, required=True, default=None, help='path to dataset')
    
    parser.add_argument('--nb_epochs', type=int, required=True, default=60, help='The number of epochs')
    parser.add_argument('--train_batchsize', type=int, required=True, default=128, help='Train batch size')
    parser.add_argument('--valid_batchsize', type=int, required=True, default=128, help='Valid batch size')
    parser.add_argument('--lr', type=float, required=True, default=0.03, help='learning rate')
    parser.add_argument('--threshold', type=float, required=True, default=0.5, help='Threshold to classify')

    parser.add_argument('--model', type=str, required=True, default="DeepSmells", help='Model to train, we have "DeepSmells" and "DeepSmell-BiLSTM"')
    parser.add_argument('--hidden_size_lstm', type=int, required=True, help='Hidden size of lstm networks')

    parser.add_argument('--tracking_dir', type=str, required=True, default="./tracking/", help='path to tracking dir')
    parser.add_argument('--result_dir', type=str, required=True, default="./result/", help='path to last result dir')

    # print(f"{'='*30}{'='*30}")
    # parser.print_help()
    # print(f"{'='*30}{'='*30}")

    args = parser.parse_args()

    return args

class Config:
    # setup DEVICE
    DEVICE = utils.device() 

    # Config train
    NB_EPOCHS = 60
    TRAIN_BS = 128
    VALID_BS = 128
    LR = 0.03
    SCALER = GradScaler()
    THRESHOLD = 0.5

    # Model
    MODEL = "LSTM"
    HIDDEN_LSTM = 1000

    # Config dir
    TRACKING_DIR = "./tracking/"
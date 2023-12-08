import torch
import torch.nn as nn
import numpy as np
import torchvision
import math

class CNN_LSTM(nn.Module):
    def __init__(self, kernel_size, input_size_lstm, hidden_size_lstm, input_dim=1, conv_dim1=16, conv_dim2=32, hidden_fc1=32, hidden_fc2=16, num_classes=1):
        super(CNN_LSTM, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_dim1, kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim1, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),

            nn.Conv1d(in_channels=conv_dim1, out_channels=conv_dim2, kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim2, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
        )

        self.lstm = nn.LSTM(input_size=input_size_lstm, hidden_size=hidden_size_lstm, batch_first=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size_lstm, hidden_fc1),
            nn.ReLU(),
            nn.Linear(hidden_fc1, hidden_fc2),
            nn.ReLU(),
            nn.Linear(hidden_fc2, num_classes),
        )
    
    def forward(self, text):
        out = self.conv_layers(text)
        out, (hidden, cell) = self.lstm(out)
        
        hidden = torch.squeeze(hidden)

        out = self.dense_layers(hidden)
        return out

class CNN_BiLSTM(nn.Module):
    def __init__(self, kernel_size, input_size_lstm, hidden_size_lstm, input_dim=1, conv_dim1=16, conv_dim2=32, hidden_fc1=48, hidden_fc2=32, num_classes=1):
        super(CNN_BiLSTM, self).__init__()
        self.hidden_size_lstm = hidden_size_lstm

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_dim1, kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim1, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),

            nn.Conv1d(in_channels=conv_dim1, out_channels=conv_dim2, kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim2, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
        )

        self.bilstm = nn.LSTM(input_size=input_size_lstm, hidden_size=input_size_lstm, batch_first=True, bidirectional=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size_lstm*2, hidden_fc1),
            nn.ReLU(),
            nn.Linear(hidden_fc1, hidden_fc2),
            nn.ReLU(),
            nn.Linear(hidden_fc2, num_classes),
        )
    
    def forward(self, text):
        out = self.conv_layers(text)
        out, (hidden, cell) = self.bilstm(out)

        hidden = torch.reshape(torch.transpose(hidden, 0, 1), (-1, self.hidden_size_lstm*2))

        out = self.dense_layers(hidden)
        return out

class CNN_LSTM_METRICS(nn.Module):
    def __init__(self, kernel_size, input_size_lstm, hidden_size_lstm, metrics_size, input_dim=1, conv_dim1=16, conv_dim2=32, hidden_fc1=32, hidden_fc2=16, num_classes=1):
        super(CNN_LSTM_METRICS, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_dim1, kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim1, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),

            nn.Conv1d(in_channels=conv_dim1, out_channels=conv_dim2, kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim2, eps=1e-07),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size),
        )

        self.lstm = nn.LSTM(input_size=input_size_lstm, hidden_size=hidden_size_lstm, batch_first=True)
        
        self.dense_layers_metrics = nn.Sequential(
            nn.Linear(metrics_size, 24),
            nn.ReLU()
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size_lstm + 24, hidden_fc1),
            nn.ReLU(),
            nn.Linear(hidden_fc1, hidden_fc2),
            nn.ReLU(),
            nn.Linear(hidden_fc2, num_classes),
        )
    
    def forward(self, text, metrics):
        out = self.conv_layers(text)
        out, (hidden, cell) = self.lstm(out)
        
        hidden = torch.squeeze(hidden).view(-1, 100)
        metrics = self.dense_layers_metrics(metrics)
        
        # print("=========)))))))))))(((((((())))))))")
        # print(f"SHAPE OF METRICS ===== {metrics.size()}")
        # print(f"=*+**)))*)**) SHAPE OF INPUT DENSE ==== {hidden.size()}")
        # print("=========)))))))))))(((((((())))))))")
        
        combine = torch.cat((hidden, metrics), axis=1)
        
        # print("=========)))))))))))(((((((())))))))")
        # print(f"SHAPE OF METRICS ===== {combine.size()}")
        
        out = self.dense_layers(combine)
        
        return out

def size_output_conv(Lin, 
                    padding=0, 
                    dilation=1, 
                    kernel_size=3, 
                    stride=1):
    return math.floor((Lin + 2 * padding - dilation*(kernel_size-1)-1)/stride + 1)

def calculate_size_lstm(input_size, kernel_size):
    # out conv 1
    out = size_output_conv(input_size, kernel_size=kernel_size)
    # out maxpool 1
    out = size_output_conv(out, kernel_size=kernel_size, stride=kernel_size)

    # out conv 2
    out = size_output_conv(out, kernel_size=kernel_size)
    # out maxpool 2
    out = size_output_conv(out, kernel_size=kernel_size, stride=kernel_size)

    return out
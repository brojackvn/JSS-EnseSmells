import gc

import torch
import numpy as np
import pandas as pd
import platform

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys 
from pathlib import Path
# Get the current directory path
current_dir = Path(__file__).parent
# sys.path.insert(0, r'/content/drive/MyDrive/LabRISE/DeepLearning-CodeSmell/DeepSmells/program/dl_models')
# Define the relative path to dl_models
relative_path_dl_models = Path("..").joinpath("dl_models")
# Add the dl_models directory to sys.path
sys.path.insert(0, str(current_dir/relative_path_dl_models))

import input_data

from sklearn.model_selection import StratifiedKFold

def write_file(file, str):
    file = open(file, mode="a+")
    file.write(str)
    file.close()

def device():
    if torch.cuda.is_available():
        print(f'[INFO] Using GPU: {torch.cuda.get_device_name()}\n')
        device = torch.device('cuda')
    else:
        print(f'\n[INFO] GPU not found. Using CPU: {platform.processor()}\n')
        device = torch.device('cpu')
    return device

# Function K fold cross validation
def get_data_pickle(data_path):
    '''
        structure of pickle file: 3 attributes
        - embedding
        - name: the identifier of the file (ex: 123456.java --> name = 123456)
        - label
    '''
    pklFile = open(data_path, 'rb')
    df = pd.read_pickle(pklFile)
    X = np.array([row for row in df['embedding']])
    y = df['label'].values

    kFold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)

    return (
        input_data.Input_data(X[train], y[train], X[test], y[test], None) for train, test in kFold.split(X, y)
    )

def get_CombineData_pickle(data_path):
    '''
        structure of pickle file: 3 attributes
        - embedding
        - metrics
        - name: the identifier of the file (ex: 123456.java --> name = 123456)
        - label
    '''
    pklFile = open(data_path, 'rb')
    df = pd.read_pickle(pklFile)
    X = np.array([row for row in df['embedding']])
    X_metrics = np.array([row for row in df['metrics']])
    y = df['label'].values

    kFold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)
    
    output = []
    for train, test in kFold.split(X, y):
        sc = StandardScaler()
        X_metrics_train = sc.fit_transform(X_metrics[train])
        X_metrics_test = sc.transform(X_metrics[test])
        output.append(input_data.Input_CombineData(X[train], X_metrics_train, y[train], X[test], X_metrics_test, y[test], None))
    
    # print(len(output))
    return output

def get_smell_and_model(path):
    smell = None
    model = None

    if "longmethod" in path.lower():
        smell = "LongMethod"
    elif "featureenvy" in path.lower():
        smell = "FeatureEnvy"
    elif "godclass" in path.lower():
        smell = "GodClass"
    elif "dataclass" in path.lower():
        smell = "DataClass"
    else:
        raise ValueError("Smell not found - CHECK FILENAME OF DATASET")
    
    if "code2seq" in path.lower():
        model = "Code2Seq"
    elif "code2vec" in path.lower():
        model = "Code2Vec"
    elif "codebert" in path.lower():
        model = "CodeBERT"
    elif "cubert" in path.lower():
        model = "CuBERT"
    elif "codet5" in path.lower():
        model = "CodeT5"
    elif "codegen" in path.lower():
        model = "CodeGen"
    elif "starcoder" in path.lower():
        model = "StarCoder"
    elif "incoder" in path.lower():
        model = "InCoder" 
    elif "tokenindexing" in path.lower():
        model = "TokenIndexing"
    else:
        raise ValueError("Model not found - CHECK FILENAME OF DATASET")
    return smell, model

# ============================= DATA TOKEN INDEXING =======================================
def get_data_token_indexing_COMBINING(data_path):
    gc.collect()

    with open(data_path, "rb") as file:
        df = pd.read_pickle(file)

    max_input_length = get_outlier_threshold(df, z=1)
    # Process give embedding length to max_input_length
    for index in range(len(df)):
        arr_size = df['embedding'][index].shape[0]

        if df['embedding'][index].shape[0] < max_input_length:
            num_zeros = max_input_length - len(df['embedding'][index])
            df['embedding'][index] = np.pad(df['embedding'][index], (0, num_zeros), 'constant')
        else:
            df['embedding'][index] = df['embedding'][index][:max_input_length]

    X = np.array([row for row in df['embedding']])
    X_metrics = np.array([row for row in df['metrics']])
    y = df['label'].values

    kFold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)
    
    output = []
    for train, test in kFold.split(X, y):
        sc = StandardScaler()
        X_metrics_train = sc.fit_transform(X_metrics[train])
        X_metrics_test = sc.transform(X_metrics[test])
        output.append(input_data.Input_CombineData(X[train], X_metrics_train, y[train], X[test], X_metrics_test, y[test], None))
    
    # print(len(output))
    return output

def get_data_token_indexing(data_path):
    gc.collect()

    with open(data_path, "rb") as file:
        df = pd.read_pickle(file)

    max_input_length = get_outlier_threshold(df, z=1)

    # Process give embedding length to max_input_length
    for index in range(len(df['embedding'])):
        arr_size = df['embedding'][index].shape[0]

        if df['embedding'][index].shape[0] < max_input_length:
            num_zeros = max_input_length - len(df['embedding'][index])
            df['embedding'][index] = np.pad(df['embedding'][index], (0, num_zeros), 'constant')
        else:
            df['embedding'][index] = df['embedding'][index][:max_input_length]

    X = np.array([row for row in df['embedding']])
    y = df['label'].values

    kFold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)

    return (
        input_data.Input_data(X[train], y[train], X[test], y[test], max_input_length) for train, test in kFold.split(X, y)
    )

def get_outlier_threshold(df, z=1):
    df_label_0 = df[df['label'] == 0].copy()
    df_label_1 = df[df['label'] == 1].copy()

    len1 = _get_outlier_threshold(df_label_1, z)
    len0 = _get_outlier_threshold(df_label_0, z)

    return max(len0, len1)

def _get_outlier_threshold(df, z):
    lengths = []
    for i in range(len(df)):
        lengths.append(df['embedding'].iloc[i].shape[0])
    return compute_max(lengths, z=z)

def compute_max(arr, dim="width", z=2):
    mn = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    final_list = [x for x in arr if (x <= mn + z * sd)]  # upper outliers removed
    rmn2 = len(arr) - len(final_list)
    print('=================================================')
    print('{} array size '.format(dim) + str(len(arr)))
    print('min {} '.format(dim) + str(min(arr, default=0)))
    print('max {} '.format(dim) + str(max(arr, default=0)))
    print('mean {} '.format(dim) + str(np.nanmean(arr)))
    print('standard deviation ' + str(np.std(arr)))
    print('median {} '.format(dim) + str(np.nanmedian(arr)))
    print('number of upper outliers removed ' + str(rmn2))
    print('max {} excluding upper outliers '.format(dim) + str(max(final_list, default=0)))
    print('=================================================')
    return max(final_list, default=0)

def _retrieve_data(path, max_len, is_c2v=False):
    input = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r',
                  errors='ignore') as file_read:
            for line in file_read:
                input_str = line.replace("\t", " ")
                if is_c2v:
                    arr = np.fromstring(input_str, dtype=np.float, sep=" ", count=max_len)
                    arr_size = len(np.fromstring(input_str, dtype=np.float, sep=" "))
                else:
                    arr = np.fromstring(input_str, dtype=np.int32, sep=" ", count=max_len)
                    arr_size = len(np.fromstring(input_str, dtype=np.int32, sep=" "))
                # We add this file only if the width is less than the outlier threshold
                if arr_size <= max_len:
                    arr[arr_size:max_len] = 0
                    input.append(arr)
    return input
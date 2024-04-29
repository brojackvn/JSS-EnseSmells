# EnseSmells: Deep ensemble and programming language models for automated code smells detection

## Dataset
```
> cd embedding-dataset
```
For efficiency during training and testing, we initially obtain embedding vectors using various techniques. Details about the processed and cleaned dataset can be found [info-dataset.md](embedding-dataset/info-dataset.md).

## Usage

Navigate to the working directory:
```
> cd EnseSmells
```

### Experiment with EnseSmells and DeepSmells
- Conduct an extensive experiment with **EnseSmells** and **DeepSmells**, utilizing various embedding techniques (token_indexing, CodeBERT, CuBERT, code2vec) within the semantic module. Ensure compatibility of `model` and `data_path` argument.
    * If `model` argument is `DeepSmells`, then `data_path` argument should be the **type 1** in [info-dataset.md](embedding-dataset/info-dataset.md).
    
    * If `model` argument is `DeepSmells_TokenIndexing`, then `data_path` argument should be the **type 2** in [info-dataset.md](embedding-dataset/info-dataset.md).
    
    * if `model` argument is `EnseSmells`, then `data_path` argument should be the **type 4** in  [info-dataset.md](embedding-dataset/info-dataset.md).
    
    * If `model` argument is `EnseSmells_TokenIndexing`, then `data_path` argument should be the **type 4** in [info-dataset.md](embedding-dataset/info-dataset.md).
- **Note**: *Each model follows its format dataset to ensure it is configured correctly.*

Run the experiment:
```
python ./program/ensesmells/main.py \
    --model "model_name" \
    --nb_epochs 85 \
    --train_batchsize 128 \
    --valid_batchsize 128 \
    --lr 0.025 \
    --threshold 0.5 \
    --hidden_size_lstm 100 \
    --data_path "path_to_pkl_file_containing_dataset" \
    --tracking_dir "path_to_folder_tracking_all_running_processes" \
    --result_dir "path_to_folder_storing_the_result"
```

### Re-run Auto-encoder Architectures
- Re-run three different auto-encoder architectures: **AE-DNN, AE-CNN** and **AE-LSTM** introduced by Sharma et al. [3]. Set up the `OUT_FOLDER` containing the result of these models on the benchmark dataset in the `smell_list` in the **autoencoder.py** file.

```
> cd program
> cd dl_models
> python autoencoder.py
```

### Re-run ML_CuBERT
- Re-run the state-of-the-art model introduced for this dataset, [ML_CuBERT](https://codeocean.com/capsule/5256791/tree/v1) [2].

### Machine Learning Classifier
- Run machine learning classifiers (NB, NN, RF, LR, CART, SVM, BP) on software metrics features with different configurations
```
> cd ml_classifier_smell
> RQ4-MLClassifierSmell.ipynb
``` 

## References
[1]. A. Ho, A. M. Bui, P. T. Nguyen, and A. Di Salle. Fusion of deep convolutional and lstm recurrent neural networks for automated detection of code smells. In Proceedings of the 27th International Conference on Evaluation and Assessment in Software Engineering, pages 229–234, 2023.

[2]. A. Kovačević, J. Slivka, D. Vidaković, K.-G. Grujić, N. Luburić, S. Prokić, and G. Sladić. Automatic detection of long method and god class code smells through neural source code embeddings. Expert Systems with Applications, 204:117607, 2022.

[3]. T. Sharma, V. Efstathiou, P. Louridas, and D. Spinellis. Code smell detection by deep direct-learning and transfer-learning. Journal of Systems and Software, 176:110936, 2021.
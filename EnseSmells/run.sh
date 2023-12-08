python ./program/ease_deepsmells/main.py \
    --model "DeepSmells_TokenIndexing_METRICS" \
    --nb_epochs 85 \
    --train_batchsize 128 \
    --valid_batchsize 128 \
    --lr 0.025 \
    --threshold 0.5 \
    --hidden_size_lstm 100 \
    --data_path "/content/drive/MyDrive/LabRISE/CodeSmellDetection/embedding-dataset/combine/GodClass/GodClass_TokenIndexing_metrics.pkl" \
    --tracking_dir "/content/drive/MyDrive/LabRISE/DeepLearning-CodeSmell/DeepSmells/tracking/combine/GodClass" \
    --result_dir "/content/drive/MyDrive/LabRISE/DeepLearning-CodeSmell/DeepSmells/results/combine"

# Here is the configure of the model
# Note: Set the configuration
# data_path: the path of the embedding file
# tracking_dir: 
# result_dir: the path of the result fi100
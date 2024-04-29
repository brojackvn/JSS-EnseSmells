python ./program/ensesmells/main.py \
    --model "DeepSmells_TokenIndexing_METRICS" \
    --nb_epochs 85 \
    --train_batchsize 128 \
    --valid_batchsize 128 \
    --lr 0.025 \
    --threshold 0.5 \
    --hidden_size_lstm 100 \
    --data_path "/Users/brojackvn/Documents/my-work/JSS-EnseSmells/embedding-dataset/combine-Embedding&Metrics/GodClass/GodClass_TokenIndexing_metrics.pkl" \
    --tracking_dir "/Users/brojackvn/Documents/my-work/JSS-EnseSmells/EnseSmells/tracking" \
    --result_dir "/Users/brojackvn/Documents/my-work/JSS-EnseSmells/EnseSmells/results"

# Here is the configure of the model
# Note: Set the configuration
# data_path: the path of the embedding file
# tracking_dir: 
# result_dir: the path of the result fi100
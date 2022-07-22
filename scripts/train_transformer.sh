set -e
name=transformer
batch_size=8
lr=5e-5
dropout=0.1
regress_layers=512,256
num_layers=4
ffn_dim=1024
nhead=4

log_dir=log_dir_path    # fix me
checkpoints_dir=checkpoints_dir_path   # fix me

target=$1
max_seq_len=$2
hidden_size=$3
feature=$4
norm_features=$5
weight_type=$6
run_idx=$7
gpu_ids=$8


cmd="python train_seed.py --dataset_mode=seq --model=transformer --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$target --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=20 --niter_decay=30 --weight_type=$weight_type
--num_threads=0 --norm_features=$norm_features --norm_method=batch
--name=$name --encoder_type=transformer
--suffix=target{target}_loss{loss_type}_{weight_type}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

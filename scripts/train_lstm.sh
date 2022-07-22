set -e
name=lstm
batch_size=8
lr=5e-5
regress_layers=256,256

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
dropout=$9


cmd="python train_seed.py --dataset_mode=seq --model=lstm --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--feature_set=$feature --target=$target --loss_type=$target --weight_type=$weight_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=30 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name
--suffix=target{target}_loss{loss_type}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}__run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

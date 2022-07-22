set -e
name=arousal_feedback
batch_size=8
lr=5e-5
dropout=0.3
regress_layers=512,256
max_seq_len=64
hidden_size=1024
num_layers=4
ffn_dim=2048
nhead=16

#weight_type: fixed, normalized_fixed, param_adjust, loss_adjust, kpi_adjust
#strategy: original, feedback, feature_feedback
#structure: original, share_bottom, only_classifier
weight_type=fixed
loss_weights=12,12,1,0.35
strategy=feature_feedback
structure=share_bottom
share_layers=3

feed_source='v au'
feed_target='a'
feed_gt_rate=0.75

log_dir=log_dir_path    # fix me
checkpoints_dir=checkpoints_dir_path   # fix me

target=$1
gpu_ids=$2

feature=feature_list # fix me
norm_features=None
loss_type=$target

suffix="target-{target}_structure-{structure}_strategy-{strategy}_share-{share_layers}_weights-{loss_weights}_type-{weight_type}_feat-{feature_set}"
if [[ $strategy == 'feedback' || $strategy == 'feature_feedback' ]];then
  suffix=$suffix"_gt-{feed_gt_rate}_fsrc-{feed_source}_ftgt-{feed_target}"
fi
suffix=$suffix"/run{run_idx}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}"

for n in {1..3};
do
run_idx=$n
cmd="python train_seed.py --dataset_mode=seq --model=mtl_transformer --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=11
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=40 --lr_policy linear_with_warmup --strategy=$strategy --structure=$structure
--num_threads=0 --norm_features=$norm_features --norm_method=trn --share_layers $share_layers
--name=$name --encoder_type=transformer
--weight_type $weight_type --gt_rate_decay
--feed_source $feed_source --feed_target $feed_target --feed_gt_rate $feed_gt_rate
--suffix=$suffix"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
done

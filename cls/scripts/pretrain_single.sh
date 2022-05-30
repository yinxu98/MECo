model=$1
dataset=$2
gpu=$3

root=.
run=${root}/scripts/pretrain_${model}.py
config=${root}/configs/${model}/
taskname=${model}_${dataset}_pretrain

CUDA_VISIBLE_DEVICES=${gpu} python ${run} ${config}/${taskname}.py
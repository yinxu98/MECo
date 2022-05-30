dataset=$1
model_type=$2 # single/multiple
test_mode=$3 # ft/lin
gpu=$4

root=.
run=${root}/scripts/test.py
config=${root}/configs/classifier
taskname=classifier_${dataset}_${model_type}_${test_mode}

CUDA_VISIBLE_DEVICES=${gpu} python ${run} ${config}/${taskname}.py
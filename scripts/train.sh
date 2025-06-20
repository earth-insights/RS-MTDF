#!/bin/bash

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['loveda', 'deepglobe', 'Potsdam']
# method: ['fixmatch_distill_with_models', 'fixmatch', 'supervised']
dataset='loveda'
method='fixmatch_distill_with_models'
exp='fixmatch_distill_with_models'
split='5'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/${split}_$(date +%Y%m%d_%H%M)

mkdir -p $save_path

torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log

#!/bin/bash
# Open-set recognition: CE vs. CE + OCCE on every benchmark, over the five known/unknown splits.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

GPU=0
MODEL="preactresnet18"
DATASETS=(mnist svhn cifar10 cifar+10 cifar+50 tiny_imagenet)
SPLITS=(0 1 2 3 4)
W_OCCE=(0.0 1.0)

for dataset in "${DATASETS[@]}"; do
  for w in "${W_OCCE[@]}"; do
    for split in "${SPLITS[@]}"; do
      python scripts/train.py --task open_set --model "$MODEL" --dataset "$dataset" --split "$split" \
        --w_ce 1.0 --w_occe "$w" --gpu "$GPU" --seed 0
    done
  done
done

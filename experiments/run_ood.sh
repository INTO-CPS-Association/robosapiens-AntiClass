#!/bin/bash
# OOD detection: CE vs. CE + OCCE on CIFAR-10 and CIFAR-100, evaluated on the near-/far-OOD sets of OpenOOD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

GPU=0
MODEL="preactresnet18"
DATASETS=(cifar10 cifar100)
SEEDS=(0 1 2)
W_OCCE=(0.0 1.0)

for dataset in "${DATASETS[@]}"; do
  for w in "${W_OCCE[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python scripts/train.py --task ood --model "$MODEL" --dataset "$dataset" \
        --w_ce 1.0 --w_occe "$w" --gpu "$GPU" --seed "$seed"
    done
  done
done

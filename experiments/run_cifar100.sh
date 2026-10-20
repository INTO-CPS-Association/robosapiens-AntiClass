#!/bin/bash
# Reproduces the CIFAR-100 / PreActResNet-18 closed-set results (Table 1):
# baseline CE vs. CE + OCCE, three seeds each.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

GPU=0
DATASET="cifar100"
MODEL="preactresnet18"
SEEDS=(0 1 2)
W_OCCE=(0.0 1.0)

for w in "${W_OCCE[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python scripts/train.py --model "$MODEL" --dataset "$DATASET" \
      --w_ce 1.0 --w_occe "$w" --gpu "$GPU" --seed "$seed"
  done
done

python scripts/plot.py

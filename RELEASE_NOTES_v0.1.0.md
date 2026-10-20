# v0.1.0 - Initial RoboSAPIENS/INTO-CPS release

First public release of AntiClass under the INTO-CPS Association.

## Included

- `OCCELoss` (one-cold cross-entropy) and baseline losses.
- PreActResNet and DenseNet architectures for CIFAR-scale inputs.
- Training script for MNIST, CIFAR-10 and CIFAR-100.
- Experiment script for CIFAR-100 / PreActResNet-18 (CE vs. CE + OCCE).
- Example TensorBoard logs and plotting script.

## Notes

Results may differ slightly (~0.1%) from the published results depending on the PyTorch version and hardware.

# Contributing

Contributions are welcome. AntiClass is maintained as research software within the RoboSAPIENS / INTO-CPS ecosystem, so changes should keep the published results reproducible.

## How to contribute

1. Open an issue describing the bug or the proposed change.
2. Create a branch from `main`.
3. Keep changes focused and update the README if command-line options or the repository layout change.
4. Open a pull request explaining the motivation and how the change was tested.

## Guidelines

- Do not commit datasets or model checkpoints.
- Report the seed and the main hyperparameters together with any new results.
- New losses should go in `losses/`, new architectures in `models/`.

## Reporting issues

Please include your OS, Python, PyTorch and CUDA versions, the exact command you ran and the error output.

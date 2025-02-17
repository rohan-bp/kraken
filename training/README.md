## Overview

This directory contains code that can be used to train and replicate the Kraken models reported in the [paper](https://neurips.cc/virtual/2024/poster/93961).
All model configurations that were discussed in the paper are available under `configs/`.

As an example, to train the 125 million parameter Kraken model with two-way parallelism on 1 GPU, you can run the following:

    python3 train.py configs/kraken_125M_2way.py

This assumes that you have updated the `data_dir` and `out_dir` config variables in the file `configs/kraken_125M_2way.py` to specify checkpoint and dataset directories.

You may want to refer to the [nanoGPT](https://github.com/karpathy/nanoGPT) repository for more information on getting the dataset set up.

For details on the precise number of GPU hours required to train each model, refer to Appendix A.1 in the paper.

## Environment

This release was tested in the following environment:
- Python 3.11.3
- PyTorch 2.2.0+cu121
- CUDA 12.6 on A100 GPUs

### Improvements to the model architecture and training recipe

If you're interested in training an improved version of the model, you may want to:

- Update the choice of position embeddings i.e., consider newer alternatives to learned positional embeddings like RoPE
- Perform a hyperparameter search for a better choice of initial learning rate
- For a fixed parameter budget, consider increasing the embedding dimension but reducing the number of layers
- Consider newer activation functions like SwiGLU in the MLP
- Vary batch size and gradient accumulation steps depending on the amount of device memory available
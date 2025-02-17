## Overview

This repository contains code that can be used to replicate the various experiments presented in [Kraken: Inherently Parallel Transformers For Efficient Multi-Device Inference](https://openreview.net/forum?id=jRtxzzk0a6).

**Kraken** is a variation of the standard Transformer architecture that is designed to complement existing tensor-parallelism schemes. The architecture introduces an innate notion of model parallelism to the Transformer layer allowing AllReduce operators to be overlapped with compute.

This repository is divided into two halves. The first, found under `training/`, contains a PyTorch implementation of the architecture as well as a simple training script and model configurations that were used in experiments.

## Replicating models

To replicate the pretrained models presented in the paper, consult the *README* in the directory `training/`.

## Citation

If you would like to cite this work, you may find the following BibTeX entry helpful:

`
@inproceedings{prabhakarkraken,
  title={{Kraken: Inherently Parallel Transformers For Efficient Multi-Device Inference}},
  author={Prabhakar, Rohan Baskar and Zhang, Hengrui and Wentzlaff, David},
  booktitle={Advances in Neural Information Processing Systems 37 (NeurIPS 2024)},
 year={2024},
}
`

### License

The applicable license can be found at the top of each source file.




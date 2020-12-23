# DeepHDR-pytorch
DeepHDR [1] (ECCV'18) re-implementation using PyTorch framework

## Introduction

This repository is the re-implementation of DeepHDR [1] using PyTorch framework. The [original repository](https://github.com/elliottwu/DeepHDR) [2] is implemented using low-version Python and Tensorflow. To make the architecture clearer and more efficient, we re-implemented it using Pytorch framework and add some basic optimizations. 

## Requirements

- PyTorch 1.4+
- Cuda version 10.1+
- OpenCV
- numpy, tqdm, scipy, etc.

## Getting Started

### Download Dataset

The Kalantari Dataset can be downloaded from https://www.robots.ox.ac.uk/~szwu/storage/hdr/kalantari_dataset.zip [2].

### Configs Modifications

You may modify the arguments in `Configs()` to satisfy your own environment, for specific arguments descriptions, see `utils/configs.py`.

### Train

```bash
export CUDA_VISIBLE_DEVICES=0
python3 train.py
```

**Note**. To generate the data patches, you need ~200 GB free storage in the folder. We will fix this data issue soon.

### Test

First, make sure that you have models (`checkpoint.tar`) under `checkpoint_dir` (which is defined in `Configs()`).

```bash
export CUDA_VISIBLE_DEVICES=0
python3 test.py
```

**Note**. `test.py` will dump the result images in `sample` folder.

## To-do

- [x] Training codes
- [x] Evaluating while training
- [x] Testing codes
- [ ] Perform the patches calculation in need to save storage
- [x] Visualizing codes
- [ ] Code re-organization
- [ ] Demo Display
- [ ] Command-line configurations support
- [ ] Pre-trained model upload

## Versions

- **v0.5 (Current version)**: Modify the codes to satisfy CUDA environment. 
- v0.4: Complete  visualization codes.
- v0.3: Complete testing codes.
- v0.2: Complete the training codes to support evaluating in training process.
- v0.1: Build the model framework and write dataset codes, training codes and utility codes.

## Reference

[1] Wu, Shangzhe, et al. "Deep high dynamic range imaging with large foreground motions." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018.

[2] elliottwu/DeepHDR repository: https://github.com/elliottwu/DeepHDR




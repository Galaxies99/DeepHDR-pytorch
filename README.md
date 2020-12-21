# DeepHDR-pytorch
DeepHDR [1] (ECCV' 18) re-implementation using PyTorch framework

### Introduction

This repository is the re-implementation of DeepHDR [1] using PyTorch framework. The [original repository](https://github.com/elliottwu/DeepHDR) is implemented using low-version Python and Tensorflow. To make the architecture clearer and more efficient, we re-implemented it using Pytorch framework and add some basic optimizations. 

### Train

```bash
export CUDA_VISIBLE_DEVICES=0
python3 train.py
```

### To-do

- [x] Training codes
- [ ] Testing codes
- [ ] Inference codes
- [ ] Code re-organization

### Reference

[1] Wu, Shangzhe, et al. "Deep high dynamic range imaging with large foreground motions." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018.


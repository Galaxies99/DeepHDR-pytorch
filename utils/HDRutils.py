# Ref: https://github.com/elliottwu/DeepHDR/
import numpy as np
import torch
import cv2

MU = 5000.  # tunemapping parameter
GAMMA = 2.2 # LDR&HDR domain transform parameter


def LDR2HDR(img, expo): # input/output 0~1
    return (((img+1)/2.)**GAMMA / expo) *2.-1


def LDR2HDR_batch(imgs, expos): # input/output 0~1
    return np.concatenate([LDR2HDR(imgs[:, :, 0:3], expos[0]),
                           LDR2HDR(imgs[:, :, 3:6], expos[1]),
                           LDR2HDR(imgs[:, :, 6:9], expos[2])], axis=2)


def HDR2LDR(imgs, expo): # input/output 0~1
    return (np.clip(((imgs+1)/2.*expo),0,1)**(1/GAMMA)) *2.-1


def transform_LDR(image, im_size=(256, 256)):
    out = image.astype(np.float32)
    out = cv2.resize(out, im_size)
    return out/127.5 - 1.


def transform_HDR(image, im_size=(256, 256)):
    out = cv2.resize(image, im_size)
    return out*2. - 1.


def tonemap(images):  # input/output 0~1
    return torch.log(1 + MU * (images + 1) / 2.) / np.log(1 + MU) * 2. - 1


def tonemap_np(images):  # input/output 0~1
    return np.log(1 + MU * (images + 1) / 2.) / np.log(1 + MU) * 2. - 1

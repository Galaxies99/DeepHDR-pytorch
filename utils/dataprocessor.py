# Ref: https://github.com/elliottwu/DeepHDR
import os
import cv2
from glob import glob
import pickle
import numpy as np
from utils.HDRutils import *


def imread(path):
    if path[-4:] == '.hdr':
        img = cv2.imread(path, -1)
    else:
        img = cv2.imread(path)/255.
    return img.astype(np.float32)[..., ::-1]


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def store_patch(h1, h2, w1, w2, in_LDRs, in_exps, ref_HDR, ref_LDRs, ref_exps, save_path, save_id):
    res = {
        'in_LDR': in_LDRs,
        'ref_LDR': ref_LDRs,
        'ref_HDR': ref_HDR,
        'in_exp': in_exps,
        'ref_exp': ref_exps,
    }
    with open(save_path + '/' + str(save_id) + '.pkl', 'wb') as pkl_file:
        pickle.dump(res, pkl_file)


def get_patch_from_file(pkl_path, pkl_id):
    with open(pkl_path + '/' + str(pkl_id) + '.pkl', 'rb') as pkl_file:
        res = pickle.load(pkl_file)
    return res


# always return RGB, float32, range 0~1
def get_image(image_path, image_size=None, is_crop=False):
    if is_crop:
        assert (image_size is not None), "the crop size must be specified"
    return transform(imread(image_path), image_size, is_crop)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def center_crop(x, image_size):
    crop_h, crop_w = image_size
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[max(0, j):min(h, j+crop_h), max(0, i):min(w, i+crop_w)], (crop_w, crop_h))


def transform(image, image_size, is_crop):
    if is_crop:
        out = center_crop(image, image_size)
    elif image_size is not None:
        out = cv2.resize(image, image_size)
    else:
        out = image
    out = out*2. - 1
    return out.astype(np.float32)


def inverse_transform(images):
    return (images + 1) / 2


# get input
def get_input(LDR_path, exp_path, ref_HDR_path):
    in_LDR_paths = sorted(glob(LDR_path))
    ns = len(in_LDR_paths)
    tmp_img = cv2.imread(in_LDR_paths[0]).astype(np.float32)
    h, w, c = tmp_img.shape
    h = h // 8 * 8
    w = w // 8 * 8

    in_exps = np.array(open(exp_path).read().split('\n')[:ns]).astype(np.float32)
    in_LDRs = np.zeros((h, w, c * ns), dtype=np.float32)
    in_HDRs = np.zeros((h, w, c * ns), dtype=np.float32)

    for i, image_path in enumerate(in_LDR_paths):
        img = get_image(image_path, image_size=[h, w], is_crop=True)
        in_LDRs[:, :, c * i:c * (i + 1)] = img
        in_HDRs[:, :, c * i:c * (i + 1)] = LDR2HDR(img, 2. ** in_exps[i])

    ref_HDR = get_image(ref_HDR_path, image_size=[h, w], is_crop=True)
    return in_LDRs, in_HDRs, in_exps, ref_HDR


def dump_sample(sample_path, img):
    img = img[0]
    h, w, _ = img.shape
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    file_path = sample_path + '/hdr.hdr'
    img = inverse_transform(img)
    img = np.einsum('ijk->jki', img)
    radiance_writer(file_path, img)

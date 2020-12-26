import cv2
import os
import argparse
from utils.dataprocessor import *
from utils.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='data/kalantari_dataset/test')
parser.add_argument('--test_path', type=str, default='')

configs = parser.parse_args()
file_path = configs.test_path
gt_path = configs.gt_path

dirs = []
for dir in os.listdir(file_path):
    if os.path.isdir(os.path.join(file_path, dir)):
        dirs.append(dir)

dirs = sorted(dirs)

psnr = PSNR()
ssim = SSIM()

total_psnr = 0
total_ssim = 0

for dir in dirs:
    gt_file = os.path.join(os.path.join(gt_path, dir), 'ref_hdr_aligned.hdr')
    my_file = os.path.join(os.path.join(file_path, dir), 'hdr.hdr')
    hdr = get_image(my_file)
    h, w, _ = hdr.shape
    hdr_gt = get_image(gt_file, [h, w], True)
    hdr_gt = inverse_transform(hdr_gt)
    hdr = inverse_transform(hdr)
    print('------------------------------------------')
    print('scene ', dir)
    cur_psnr = psnr(hdr, hdr_gt)
    cur_ssim = ssim(hdr, hdr_gt)
    print('PSNR:', cur_psnr)
    print('SSIM:', cur_ssim)
    total_psnr += cur_psnr
    total_ssim += cur_ssim

print('******************************************')
print('Final Report:')
print('  Average PSNR: ', total_psnr / len(dirs))
print('  Average SSIM: ', total_ssim / len(dirs))

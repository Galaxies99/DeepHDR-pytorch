import os
import cv2
import glob
import torch
import numpy as np
import pickle
from tqdm import tqdm
from utils.HDRutils import *
from torch.utils.data import Dataset


def store_patch(h1, h2, w1, w2, in_LDRs, in_exps, ref_HDR, ref_LDRs, ref_exps, save_path, save_id):
    in_LDRs_patch = in_LDRs[h1:h2, w1:w2, :]
    in_LDRs_patch_1 = in_LDRs_patch[:, :, 2::-1]
    in_LDRs_patch_2 = in_LDRs_patch[:, :, 5:2:-1]
    in_LDRs_patch_3 = in_LDRs_patch[:, :, 8:5:-1]
    in_LDRs_patch = np.concatenate([in_LDRs_patch_1, in_LDRs_patch_2, in_LDRs_patch_3], axis=2)
    ref_HDR_patch = ref_HDR[h1:h2, w1:w2, ::-1]
    ref_LDRs_patch = ref_LDRs[h1:h2, w1:w2, :]
    ref_LDRs_patch_1 = ref_LDRs_patch[:, :, 2::-1]
    ref_LDRs_patch_2 = ref_LDRs_patch[:, :, 5:2:-1]
    ref_LDRs_patch_3 = ref_LDRs_patch[:, :, 8:5:-1]
    ref_LDRs_patch = np.concatenate([ref_LDRs_patch_1, ref_LDRs_patch_2, ref_LDRs_patch_3], axis=2)

    res = {
        'in_LDR': in_LDRs_patch,
        'ref_LDR': ref_LDRs_patch,
        'ref_HDR': ref_HDR_patch,
        'in_exp': in_exps,
        'ref_exp': ref_exps,
    }

    with open(save_path + '/' + str(save_id) + '.pkl', 'wb') as pkl_file:
        pickle.dump(res, pkl_file)


def get_patch(pkl_path, pkl_id):
    with open(pkl_path + '/' + str(pkl_id) + '.pkl', 'rb') as pkl_file:
        res = pickle.load(pkl_file)
    return res


class KalantariDataset(Dataset):
    def __init__(self, configs,
                 input_name= 'input_*_aligned.tif',
                 ref_name = 'ref_*_aligned.tif',
                 input_exp_name = 'input_exp.txt',
                 ref_exp_name = 'ref_exp.txt',
                 ref_hdr_name = 'ref_hdr_aligned.hdr'):
        super().__init__()
        filepath = os.path.join(configs.data_path, 'train')
        self.scene_dirs = [scene_dir for scene_dir in os.listdir(filepath)
                            if os.path.isdir(os.path.join(filepath, scene_dir))]
        self.scene_dirs = sorted(self.scene_dirs)
        self.num_scenes = len(self.scene_dirs)

        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.num_shots = configs.num_shots
        self.batch_size = configs.batch_size
        self.patch_path = configs.patch_dir
        self.count = len(os.listdir(self.patch_path))

        if self.count == 0:
            for i, scene_dir in tqdm(enumerate(self.scene_dirs)):
                cur_scene_dir = os.path.join(filepath, scene_dir)
                in_LDR_paths = sorted(glob.glob(os.path.join(cur_scene_dir, input_name)))
                tmp_img = cv2.imread(in_LDR_paths[0]).astype(np.float32)
                h, w, c = tmp_img.shape

                in_LDRs = np.zeros((h, w, c * self.num_shots))
                for j, in_LDR_path in enumerate(in_LDR_paths):
                    in_LDRs[:, :, j * c:(j + 1) * c] = cv2.imread(in_LDR_path).astype(np.float32)
                in_LDRs = in_LDRs.astype(np.float32)
                in_exps_path = os.path.join(cur_scene_dir, input_exp_name)
                in_exps = np.array(open(in_exps_path).read().split('\n')[:self.num_shots]).astype(np.float32)
                ref_HDR = cv2.imread(os.path.join(cur_scene_dir, ref_hdr_name), -1).astype(np.float32)

                ref_LDR_paths = sorted(glob.glob(os.path.join(cur_scene_dir, ref_name)))
                ref_LDRs = np.zeros((h, w, c * self.num_shots))
                for j, ref_LDR_path in enumerate(ref_LDR_paths):
                    ref_LDRs[:, :, j * c:(j + 1) * c] = cv2.imread(ref_LDR_path).astype(np.float32)
                ref_LDRs = ref_LDRs.astype(np.float32)
                ref_exps_path = os.path.join(cur_scene_dir, ref_exp_name)
                ref_exps = np.array(open(ref_exps_path).read().split('\n')[:self.num_shots]).astype(np.float32)

                # Cut the images into several patches
                # Store patches into files to save memory.
                for _h in range(0, h - self.patch_size + 1, self.patch_stride):
                    for _w in range(0, w - self.patch_size + 1, self.patch_stride):
                        store_patch(_h, _h + self.patch_size, _w, _w + self.patch_size, in_LDRs, in_exps,
                                    ref_HDR, ref_LDRs, ref_exps, self.patch_path, self.count)
                        self.count += 1
                if h % self.patch_size:
                    for _w in range(0, w - self.patch_size + 1, self.patch_stride):
                        store_patch(h - self.patch_size, h, _w, _w + self.patch_size, in_LDRs, in_exps,
                                    ref_HDR, ref_LDRs, ref_exps, self.patch_path, self.count)
                        self.count += 1
                if w % self.patch_size:
                    for _h in range(0, h - self.patch_size + 1, self.patch_stride):
                        store_patch(_h, _h + self.patch_size, w - self.patch_size, w, in_LDRs, in_exps,
                                    ref_HDR, ref_LDRs, ref_exps, self.patch_path, self.count)
                        self.count += 1
                if h % self.patch_size and w % self.patch_size:
                    store_patch(h - self.patch_size, h, w - self.patch_size, w, in_LDRs, in_exps,
                                ref_HDR, ref_LDRs, ref_exps, self.patch_path, self.count)
                    self.count += 1
        print('Finish preparing Data!')

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        distortions = np.random.uniform(0.0, 1.0, 2)

        data = get_patch(self.patch_path, index)
        in_LDR = data['in_LDR']
        ref_LDR = data['ref_LDR']
        ref_HDR = data['ref_HDR']
        in_exp = data['in_exp']
        ref_exp = data['ref_exp']

        # Horizontal flip
        if distortions[0] < 0.5:
            in_LDR = np.flip(in_LDR, axis=1)
            ref_LDR = np.flip(ref_LDR, axis=1)
            ref_HDR = np.flip(ref_HDR, axis=1)

        # Rotation
        k = int(distortions[1] * 4 + 0.5)
        in_LDR = np.rot90(in_LDR, k)
        ref_LDR = np.rot90(ref_LDR, k)
        ref_HDR = np.rot90(ref_HDR, k)
        in_exp = 2 ** in_exp
        ref_exp = 2 ** ref_exp

        in_HDR = LDR2HDR_batch(in_LDR, in_exp)

        # In pytorch, channels is in axis=1, so ijk -> kij
        in_LDR = np.einsum("ijk->kij", in_LDR)
        ref_LDR = np.einsum("ijk->kij", ref_LDR)
        in_HDR = np.einsum("ijk->kij", in_HDR)
        ref_HDR = np.einsum("ijk->kij", ref_HDR)
        return in_LDR.copy().astype(np.float32), ref_LDR.copy().astype(np.float32), \
               in_HDR.copy().astype(np.float32), ref_HDR.copy().astype(np.float32), \
               in_exp.copy().astype(np.float32), ref_exp.copy().astype(np.float32)

import os
import cv2
from glob import glob
import numpy as np
from utils.dataset import *
from utils.HDRutils import *
from torch.utils.data import Dataset



class KalantariDataset(Dataset):
    def __init__(self, configs,
                 input_name= 'input_*_aligned.tif',
                 ref_name = 'ref_*_aligned.tif',
                 input_exp_name = 'input_exp.txt',
                 ref_exp_name = 'ref_exp.txt',
                 ref_hdr_name = 'ref_hdr_aligned.hdr'):
        super().__init__()
        print('====> Start preparing training data.')
        filepath = os.path.join(configs.data_path, 'train')
        self.scene_dirs = [scene_dir for scene_dir in os.listdir(filepath)
                            if os.path.isdir(os.path.join(filepath, scene_dir))]
        self.scene_dirs = sorted(self.scene_dirs)
        self.num_scenes = len(self.scene_dirs)

        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.num_shots = configs.num_shots
        self.patch_path = configs.patch_dir
        self.count = len(os.listdir(self.patch_path))

        if self.count == 0:
            for i, scene_dir in enumerate(self.scene_dirs):
                cur_scene_dir = os.path.join(filepath, scene_dir)
                in_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, input_name)))
                tmp_img = cv2.imread(in_LDR_paths[0]).astype(np.float32)
                h, w, c = tmp_img.shape
                in_LDRs = np.zeros((h, w, c * self.num_shots))
                for j, in_LDR_path in enumerate(in_LDR_paths):
                    in_LDRs[:, :, j * c:(j + 1) * c] = cv2.imread(in_LDR_path).astype(np.float32) / 255.0
                in_LDRs = in_LDRs.astype(np.float32)
                in_exps_path = os.path.join(cur_scene_dir, input_exp_name)
                in_exps = np.array(open(in_exps_path).read().split('\n')[:self.num_shots]).astype(np.float32)
                ref_HDR = cv2.imread(os.path.join(cur_scene_dir, ref_hdr_name), -1).astype(np.float32)
                ref_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, ref_name)))
                ref_LDRs = np.zeros((h, w, c * self.num_shots))
                for j, ref_LDR_path in enumerate(ref_LDR_paths):
                    ref_LDRs[:, :, j * c:(j + 1) * c] = cv2.imread(ref_LDR_path).astype(np.float32) / 255.0
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
        print('====> Finish preparing training data!')

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        distortions = np.random.uniform(0.0, 1.0, 2)

        data = get_patch_from_file(self.patch_path, index)
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


class KalantariTestDataset(Dataset):
    def __init__(self, configs,
                 input_name= 'input_*_aligned.tif',
                 input_exp_name = 'input_exp.txt',
                 ref_hdr_name = 'ref_hdr_aligned.hdr'):
        super().__init__()
        print('====> Start preparing testing data.')
        self.filepath = os.path.join(configs.data_path, 'test')
        self.scene_dirs = [scene_dir for scene_dir in os.listdir(self.filepath)
                            if os.path.isdir(os.path.join(self.filepath, scene_dir))]
        self.scene_dirs = sorted(self.scene_dirs)
        self.num_scenes = len(self.scene_dirs)
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.num_shots = configs.num_shots
        self.sample_path = configs.sample_dir
        self.input_name = input_name
        self.input_exp_name = input_exp_name
        self.ref_hdr_name = ref_hdr_name
        print('====> Finish preparing testing data!')

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, index):
        scene_dir = self.scene_dirs[index]
        scene_path = os.path.join(self.filepath, scene_dir)
        sample_path = os.path.join(self.sample_path, scene_dir)
        LDR_path = os.path.join(scene_path, self.input_name)
        exp_path = os.path.join(scene_path, self.input_exp_name)
        ref_HDR_path = os.path.join(scene_path, self.ref_hdr_name)
        in_LDRs, in_HDRs, in_exps, ref_HDRs = get_input(LDR_path, exp_path, ref_HDR_path)
        in_LDRs = np.einsum("ijk->kij", in_LDRs)
        in_HDRs = np.einsum("ijk->kij", in_HDRs)
        ref_HDRs = np.einsum("ijk->kij", ref_HDRs)
        return sample_path, \
               in_LDRs.copy().astype(np.float32), \
               in_HDRs.copy().astype(np.float32), \
               in_exps.copy().astype(np.float32), \
               ref_HDRs.copy().astype(np.float32)


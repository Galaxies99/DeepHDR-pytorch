import os
import cv2
from glob import glob
import numpy as np
from utils.dataprocessor import *
from utils.HDRutils import *
from torch.utils.data import Dataset


class KalantariDataset(Dataset):
    def __init__(self, configs,
                 input_name='input_*_aligned.tif',
                 ref_name='ref_*_aligned.tif',
                 input_exp_name='input_exp.txt',
                 ref_exp_name='ref_exp.txt',
                 ref_hdr_name='ref_hdr_aligned.hdr'):
        super().__init__()
        print('====> Start preparing training data.')

        # Some basic information
        self.filepath = os.path.join(configs.data_path, 'train')
        self.scene_dirs = [scene_dir for scene_dir in os.listdir(self.filepath)
                            if os.path.isdir(os.path.join(self.filepath, scene_dir))]
        self.scene_dirs = sorted(self.scene_dirs)
        self.num_scenes = len(self.scene_dirs)
        self.patch_size = configs.patch_size
        self.image_size = configs.image_size
        self.patch_stride = configs.patch_stride
        self.num_shots = configs.num_shots
        self.input_name = input_name
        self.ref_name = ref_name
        self.input_exp_name = input_exp_name
        self.ref_exp_name = ref_exp_name
        self.ref_hdr_name = ref_hdr_name
        self.total_count = 0
        # Count the number of patches in each trainning image
        self.count = []
        for i, scene_dir in enumerate(self.scene_dirs):
            cur_scene_dir = os.path.join(self.filepath, scene_dir)
            in_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, input_name)))
            tmp_img = get_image(in_LDR_paths[0]).astype(np.float32)
            h, w, c = tmp_img.shape
            if h < self.patch_size[0] or w < self.patch_size[1]:
                raise AttributeError('The size of some trainning images are smaller than the patch size.')
            h_count = np.ceil(h / self.patch_stride)
            w_count = np.ceil(w / self.patch_stride)
            self.count.append(h_count * w_count)
            self.total_count = self.total_count + h_count * w_count
        self.count = np.array(self.count).astype(int)
        self.total_count = int(self.total_count)

        print('====> Finish preparing training data!')

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        # Find the corresponding image
        idx_beg = 0
        cur_scene_dir = ""
        scene_idx = -1
        scene_posidx = -1
        for i, scene_dir in enumerate(self.scene_dirs):
            idx_end = idx_beg + self.count[i]
            if idx_beg <= index < idx_end:
                cur_scene_dir = os.path.join(self.filepath, scene_dir)
                scene_idx = i
                scene_posidx = index - idx_beg
                break
            idx_beg = idx_end
        if scene_idx == -1:
            raise ValueError('Index out of bound')

        in_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, self.input_name)))
        tmp_img = get_image(in_LDR_paths[0])
        h, w, c = tmp_img.shape

        # Count the indices of h and w
        h_count = np.ceil(h / self.patch_stride)
        w_count = np.ceil(w / self.patch_stride)
        h_idx = int(scene_posidx / w_count)
        w_idx = int(scene_posidx - h_idx * w_count)

        # Count the up, down, left, right of the patch
        h_up = h_idx * self.patch_stride
        h_down = h_idx * self.patch_stride + self.patch_size[0]
        if h_down > h:
            h_up = h - self.patch_size[0]
            h_down = h

        w_left = w_idx * self.patch_stride
        w_right = w_idx * self.patch_stride + self.patch_size[1]
        if w_right > w:
            w_left = w - self.patch_size[1]
            w_right = w

        # Get the input images
        in_LDR = np.zeros((self.patch_size[0], self.patch_size[1], c * self.num_shots))
        for j, in_LDR_path in enumerate(in_LDR_paths):
            in_LDR[:, :, j * c:(j + 1) * c] = get_image(in_LDR_path)[h_up:h_down, w_left:w_right, :]
        in_LDR = np.array(in_LDR).astype(np.float32)

        in_exp_path = os.path.join(cur_scene_dir, self.input_exp_name)
        in_exp = np.array(open(in_exp_path).read().split('\n')[:self.num_shots]).astype(np.float32)

        ref_HDR = get_image(os.path.join(cur_scene_dir, self.ref_hdr_name))[h_up:h_down, w_left:w_right, :]

        ref_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, self.ref_name)))
        ref_LDR = np.zeros((self.patch_size[0], self.patch_size[1], c * self.num_shots))
        for j, ref_LDR_path in enumerate(ref_LDR_paths):
            ref_LDR[:, :, j * c:(j + 1) * c] = get_image(ref_LDR_path)[h_up:h_down, w_left:w_right, :]
        ref_LDR = np.array(ref_LDR).astype(np.float32)

        ref_exp_path = os.path.join(cur_scene_dir, self.ref_exp_name)
        ref_exp = np.array(open(ref_exp_path).read().split('\n')[:self.num_shots]).astype(np.float32)

        # Make some random transformation.
        distortions = np.random.uniform(0.0, 1.0, 2)
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
        in_LDR, in_HDRs, in_exp, ref_HDR = get_input(LDR_path, exp_path, ref_HDR_path)
        in_LDR = np.einsum("ijk->kij", in_LDR)
        in_HDRs = np.einsum("ijk->kij", in_HDRs)
        ref_HDR = np.einsum("ijk->kij", ref_HDR)
        return sample_path, \
               in_LDR.copy().astype(np.float32), \
               in_HDRs.copy().astype(np.float32), \
               in_exp.copy().astype(np.float32), \
               ref_HDR.copy().astype(np.float32)


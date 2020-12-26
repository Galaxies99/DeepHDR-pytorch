import os

class Configs(object):
    def __init__(self, data_path='data',
                 **kwargs):
        self.data_path = data_path
        self.epoch = kwargs.get('epoch', 50)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.load_size = kwargs.get('load_size', 250)
        self.patch_size = kwargs.get('patch_size', (256, 256))
        self.image_size = kwargs.get('image_size', (256, 256))
        self.patch_stride = kwargs.get('patch_stride', 64)
        self.patch_dir = kwargs.get('patch_dir', 'patches')
        self.batch_size = kwargs.get('batch_size', 32)
        self.c_dim = kwargs.get('c_dim', 3)
        self.num_shots = kwargs.get('num_shots', 3)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoint')
        self.sample_dir = kwargs.get('sample_dir', 'samples')
        self.log_dir = kwargs.get('log_dir', 'logs')
        self.multigpu = kwargs.get('multigpu', False)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.patch_dir):
            os.makedirs(self.patch_dir)

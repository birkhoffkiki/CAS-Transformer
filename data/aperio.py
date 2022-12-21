
"""
aperio_hamamastu dataset, 
reference: https://github.com/khtao/StainNet
"""
from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os


class AperioDataset(Dataset):
    def __init__(self, args, phase=None, random_seed=0):
        self.random_seed = random_seed
        random.seed(random_seed)
        args = args.DATA
        self.args = args
        self.phase = phase if phase == 'train' else 'test'   # The behaviour of val and test is same
        self.data_root = args.DATA_PATH
        path = os.path.join(self.data_root, self.phase, 'aperio')
        self.image_names = [i for i in os.listdir(path) if '.png' in i]

    def __len__(self):
        return len(self.image_names)

    def read_img(self, name):
        he_path = os.path.join(self.data_root, self.phase, 'aperio', name)
        ihc_path = os.path.join(self.data_root, self.phase, 'hamamatsu', 'H'+name[1:]) # val is from train
        he = Image.open(he_path)
        ihc = Image.open(ihc_path)
        return he, ihc

    def multi_scale(self, img):
        """Different from the silico datasets
        """
        input_small = functional.resize(img, self.args.IMG_SIZE[0])
        input_big = functional.resize(img, self.args.IMG_SIZE[2])
        return input_small, img, input_big

    def __getitem__(self, item):
        name = self.image_names[item]
        he, ihc = self.read_img(name)
        # flip
        if self.phase == 'train':
            flip_func = self.flip()
            he = flip_func(he)
            ihc = flip_func(ihc)
        he_small, he_middle, he_big = self.multi_scale(he)
        ihc_small, ihc_middle, ihc_big = self.multi_scale(ihc)
        he_data = [functional.to_tensor(i) for i in [he_small, he_middle, he_big]]
        ihc_data = [functional.to_tensor(i) for i in [ihc_small, ihc_middle, ihc_big]]
        data = {
            'he': he_data,
            'ihc': ihc_data,
            'name': name,
        }
        return data

    @staticmethod
    def flip():
        h = random.random()
        v = random.random()        
        def func(input_data):
            input_data = functional.hflip(input_data) if h > 0.5 else input_data
            input_data = functional.vflip(input_data) if v > 0.5 else input_data
            return input_data
        return func
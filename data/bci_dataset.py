"""
About data, please see https://github.com/bupt-ai-cz/BCI
"""
from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os


class BCIDataset(Dataset):
    def __init__(self, args, phase=None, random_seed=0):
        self.random_seed = random_seed
        random.seed(random_seed)
        args = args.DATA
        self.args = args
        self.phase = phase if phase == 'train' else 'val'   # The behaviour of val and test is same
        self.data_root = args.DATA_PATH
        path = os.path.join(self.data_root, self.phase, 'HE')
        image_names = os.listdir(path)
        if self.phase == 'train':
            # random crop images
            self.image_names = [(name, (0, 0)) for name in image_names]
        else:
            self.image_names = []
            for name in image_names:
                for i in range(1024//args.IMG_SIZE[1]):
                    for j in range(1024//args.IMG_SIZE[1]):
                        self.image_names.append((name, (i*args.IMG_SIZE[1], j*args.IMG_SIZE[1])))

    def __len__(self):
        return len(self.image_names)

    def read_img(self, item):
        name, dtype = item
        size = self.args.IMG_SIZE[1]
        he_path = os.path.join(self.data_root, self.phase, 'HE', name)
        ihc_path = os.path.join(self.data_root, self.phase, 'IHC', name) # val is from train
        he = Image.open(he_path)
        ihc = Image.open(ihc_path)
        if dtype is None:
            x = random.randint(0, 1024-size)
            y = random.randint(0, 1024-size)
        else:
            x, y = dtype
        he = functional.crop(he, x, y, size, size)
        ihc = functional.crop(ihc, x, y, size, size)
        name = '{}_{}_{}'.format(x, y, name)
        return he, ihc, name

    def multi_scale(self, img):
        """Different from the silico datasets
        """
        input_small = functional.resize(img, self.args.IMG_SIZE[0])
        input_big = functional.resize(img, self.args.IMG_SIZE[2])
        return input_small, img, input_big

    def __getitem__(self, item):
        name = self.image_names[item]
        he, ihc, name = self.read_img(name)
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
            'info': self.image_names[item][0],
            'xy': self.image_names[item][1],
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
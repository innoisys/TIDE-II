import os
import numpy as np

from PIL import Image
from re import split, compile
from tensorflow.keras.utils import Sequence


def list_filenames(data_path, img_extension='png', filename_prefix=None):
    if filename_prefix is None:
        files_list = [file for file in os.listdir(data_path) if file.endswith(img_extension)]
    else:
        files_list = [file for file in os.listdir(data_path) if file.endswith(img_extension) and file.startswith(filename_prefix)]

    files_list.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in split(compile(r'(\d+)'), l)])
    files_list = [os.path.join(data_path, file) for file in files_list]
    print('Found {} files in {}'.format(len(files_list), data_path))
    return files_list


class Dataset(Sequence):
    def __init__(self, file_list, batch_size=32, crop_dim=None, resize_dim=None, shuffle=True, mode='RGB'):
        self.files_list = file_list
        self.batch_size = batch_size

        self.crop_dim = crop_dim
        self.resize_dim = resize_dim
        self.shuffle = shuffle
        self.on_epoch_end()

        self.mode=mode

    def __len__(self):
        return int(np.ceil(len(self.files_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.files_list[idx * self.batch_size : (idx + 1) * self.batch_size]
        images = [self.load_images(f) for f in batch_files]
        return np.stack(images)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files_list)

    @staticmethod
    def center_crop(image, crop_dim):
        h, w = image.size
        crop_h, crop_w = crop_dim

        top = max(0, (w - crop_w) // 2)
        left = max(0, (h - crop_h) // 2)
        right = min(h - 0, (h + crop_h) // 2)
        bottom = min(w - 0, (w + crop_w) // 2)

        return image.crop((left, top, right, bottom))

    def load_images(self, filepath):
        if self.mode=='RGB':
            image = Image.open(filepath).convert('RGB')
        else:
            image = Image.open(filepath)
        if self.crop_dim:
            image = self.center_crop(image, crop_dim=self.crop_dim)
        if self.resize_dim:
            image = image.resize(self.resize_dim)

        image = np.array(image).astype(np.float32)
        image = image / 255.0
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        return image

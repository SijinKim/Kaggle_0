from glob import glob
import os

from skimage import io
from torch.utils.data import Dataset


DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data'
)


class PhotoDataset(Dataset):
    def __init__(self):
        super(PhotoDataset, self).__init__()
        self.paths = glob(os.path.join(DATA_ROOT, 'photo_jpg', '*'))

    def __getitem__(self, x):
        return io.imread(self.paths[x])

    def __len__(self):
        return len(self.paths)


class MonetDataset(Dataset):
    def __init__(self):
        super(MonetDataset, self).__init__()
        self.paths = glob(os.path.join(DATA_ROOT, 'monet_jpg', '*.jpg'))

    def __getitem__(self, x):
        return io.imread(self.paths[x])

    def __len__(self):
        return len(self.paths)

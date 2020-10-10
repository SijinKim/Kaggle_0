import os
from glob import glob

from torch.utils.data import Dataset
from skimage import io


class PhotoDataset(Dataset):
    def __init__(self):
        super(PhotoDataset, self).__init__()
        DATA_ROOT = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data'
        )
        self.photo_files = glob(os.path.join(DATA_ROOT, 'photo_jpg', '*'))

    def __getitem__(self, x):
        return io.imread(self.photo_files[x])

    def __len__(self):
        return len(self.photo_files)

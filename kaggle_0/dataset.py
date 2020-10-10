import os
import glob
import skimage.io as io
from torch.utils.data import Dataset


class MonetDataset(Dataset):
    def __init__(self):
        super(MonetDataset, self).__init__()
        DATA_ROOT = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data'
        )
        self.path = glob.glob(os.path.join(DATA_ROOT, 'monet_jpg', '*.jpg'))

    def __getitem__(self, x):
        return io.imread(self.path[x])

    def __len__(self):
        return len(self.path)

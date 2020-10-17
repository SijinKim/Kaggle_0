import unittest

from kaggle_0.dataset import PhotoDataset
from kaggle_0.dataset import MonetDataset


class TestPhotoDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = PhotoDataset()

    def test_shape_of_loaded_image_array(self):
        data = self.dataset[0]
        self.assertEqual(data.shape, (256, 256, 3))

    def test_number_of_images(self):
        self.assertEqual(len(self.dataset), 7038)


class TestMonetDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = MonetDataset()

    def test_shape_of_loaded_image_array(self):
        data = self.dataset[0]
        self.assertEqual(data.shape, (256, 256, 3))

    def test_number_of_images(self):
        n_data = len(self.dataset)
        self.assertEqual(n_data, 300)

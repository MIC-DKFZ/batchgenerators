import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class DummyDL(SlimDataLoaderBase):
    def generate_train_batch(self):
        a = np.random.randn(1000, 1000)
        a_squared = a @ a
        return a_squared


mt = MultiThreadedAugmenter(DummyDL(None, None), None, 8, 2, None, False)

for i in range(200):
    a = next(mt)

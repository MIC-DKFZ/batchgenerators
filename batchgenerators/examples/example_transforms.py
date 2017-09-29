#replace these three lines with your own stuff
from Datasets.Brain_Tumor_450k_new import load_dataset_2, BatchGenerator3D_random_sampling
dataset = load_dataset_2()
b = BatchGenerator3D_random_sampling(dataset, 2, None, False, (128, 128, 128), None, False)

data_dict = b.next()

from batchgenerators.transforms.spatials_transforms import SpatialTransform, Mirror
from batchgenerators.transforms.color_transforms import GammaTransform

s = SpatialTransform((100, 100, 100), 30, True)
m = Mirror((2, 3, 4))
g = GammaTransform()

from batchgenerators.transforms.abstract_transform import Compose, RndTransform

rd_s = RndTransform(s)

c = Compose([rd_s, m, g])

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

mth = MultiThreadedAugmenter(b, c, 8, 2, None)
mth.restart()

d = mth.next()
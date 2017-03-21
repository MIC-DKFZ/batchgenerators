import numpy as np
import matplotlib.pyplot as plt
import Datasets.Brain_Tumor_450k as dataset
from Datasets.Brain_Tumor_450k import BatchGenerator3D, get_patch_locs
from MultiThreadedGenerator import MultiThreadedGenerator
from DataAugmentationGenerators import ultimate_transform_generator, center_crop_generator

data = dataset.load_patients_450k_simple_normalization()
patch_locs = get_patch_locs(data, 1, 1, 1, 128)

a = BatchGenerator3D(data, 1, None, False, patch_locs)
a = center_crop_generator(a, 96)
a = ultimate_transform_generator(a, True, (0, 1500), (8, 12), True, do_scale=True, scale=(0.5, 2))
a = MultiThreadedGenerator(a, 3, 1, seeds=None)
a._start()

b = a.next()
plt.subplot(1, 2, 1)
plt.imshow(b['data'][0, 1, 32], cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(b['seg'][0, 0, 32], cmap="gray")



ctr = 0
for data_dict in a:
    ctr += 1
    if ctr > 10:
        break
print "generated ", ctr, " batches"
import numpy as np
import matplotlib.pyplot as plt
import Datasets.Brain_Tumor_450k as dataset
from Datasets.Brain_Tumor_450k import BatchGenerator3D, get_patch_locs
from MultiThreadedGenerator import MultiThreadedGenerator
from DeepLearningBatchGeneratorUtils.DataAugmentationGenerators import rotation_and_elastic_transform_generator

data = dataset.load_patients_450k_simple_normalization()
patch_locs = get_patch_locs(data, 1, 1, 1, 128)

a = BatchGenerator3D(data, 2, 2, False, patch_locs)
a = rotation_and_elastic_transform_generator(a, 1000, 10)
a = MultiThreadedGenerator(a, 10, 1, seeds=[1234]*10)
ctr = 0
for data_dict in a:
    ctr += 1
    if ctr > 10:
        break
print "generated ", ctr, " batches"
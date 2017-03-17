from DeepLearningBatchGeneratorUtils.DataGeneratorBase import BatchGeneratorBase
import numpy as np

class MockBatchGenerator(BatchGeneratorBase):

    def generate_train_batch(self):

        #Sample randomly from data
        idx = np.random.choice(self._data[0].shape[0], self.BATCH_SIZE, False, None)
        # copy data to ensure that we are not modifying the original dataset with subsequeng augmentation techniques!
        x = np.array(self._data[0][idx])
        y = np.array(self._data[1][idx])

        data_dict = {"data": x,
                     "seg": y}
        return data_dict
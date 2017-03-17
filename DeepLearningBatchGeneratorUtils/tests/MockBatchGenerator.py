from DeepLearningBatchGeneratorUtils.DataGeneratorBase import BatchGeneratorBase
import numpy as np

class MockBatchGenerator(BatchGeneratorBase):

    def generate_train_batch(self):

        x = np.array(self._data[0][0:self.BATCH_SIZE])
        y = np.array(self._data[1][0:self.BATCH_SIZE])

        data_dict = {"data": x,
                     "seg": y}
        return data_dict
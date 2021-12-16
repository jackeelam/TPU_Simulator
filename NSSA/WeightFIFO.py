import numpy as np
from queue import Queue

class WeightFIFO:
    """
    TPU Weight FIFO Fetcher

    Stores all the weights and performs the diagonal pipelining of the weights 
    through the systolic array MMU
    """
    def __init__(self, weights_size):
        self.weights_size = weights_size
        self.weights = []
        self.weight_buffer = []
        for i in range(self.weights_size):
            self.weight_buffer.append(Queue()) # init with empty queues

    def add_weights(self, weights):
        """
        Add layer weights to the weight FIFO
        """
        weights = weights.T
        offset = 0 if len(self.weights) == 0 else np.clip(weights.shape[1] - self.weights[-1].shape[1], 0, None)
        for col in range(self.weights_size):
            # pad with traingle of zeroes, along with offset padding
            for i in range(np.clip(col - self.weight_buffer[col].qsize(), 0, None) + offset):
                self.weight_buffer[col].put(0)

            # insert weights, or fill with zeros
            buffer = np.zeros(weights[0].shape[0])
            if col < len(weights):
                buffer = weights[col]
            for element in buffer:
                self.weight_buffer[col].put(element)

            for i in range(weights.shape[0] + 1):
                self.weight_buffer[col].put(0) # NSSA pattern to pad pipeline with 0s

        self.weights.append(weights)
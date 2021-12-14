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
        self.weight_buffer = []
        for i in range(self.weights_size):
            self.weight_buffer.append(Queue()) # init with empty queues

    def add_weights(self, weights, offset):
        """
        Add layer weights to the weight FIFO
        """
        for col in range(self.weights_size):
            # pad with traingle of zeroes, along with offset padding
            for i in range(np.clip(col - self.weight_buffer[col].qsize(), 0, None) + offset):
                self.weight_buffer[col].put(0)

            # insert input, or fill with zeros
            buffer = np.zeros(weights[0].shape[0])
            if col < len(input):
                buffer = weights[col]
            for element in buffer:
                self.weight_buffer[col].put(element)

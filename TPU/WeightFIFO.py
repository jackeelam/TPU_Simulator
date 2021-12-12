import numpy as np
from queue import Queue

# Weight FIFO buffer
class WeightFIFO:
    def __init__(self):
        self.weight_buffer = Queue()

    # Add layer weights to weight FIFO buffer
    def add_weights(self, weights):
        self.weight_buffer.put(weights)

    # Remove weights from weight buffer given dimensions
    def get_weights(self):
        return self.weight_buffer.get()

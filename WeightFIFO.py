import numpy as np
from queue import Queue

# Weight FIFO buffer
class WeightFIFO:
    def __init__(self, weights):
        self.weight_buffer = Queue()
        self.add_weights(weights)

    # Add layer weights to weight FIFO buffer
    def add_weights(self, weights):
        for layer in range(len(weights)):
            self.weight_buffer.put(layer)

    # Remove weights from weight buffer given dimensions
    def get_weights(self):
        return self.weight_buffer.get()

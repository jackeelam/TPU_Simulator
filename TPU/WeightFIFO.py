import numpy as np
from queue import Queue

# Weight FIFO buffer
class WeightFIFO:
    def __init__(self, mmu, inputs_data):
        self.inputs = inputs_data
        self.weight_buffer = Queue()

        self.mmu = mmu

        self.current_input_size = 0
        self.flood_weights = np.zeros(self.mmu.shape)
        self.flood_layers = []
        self.flood_indices = []

        self.wait_cycles = 0

    def initialize_flood(self):
        weights = self.weight_buffer.get()
        self.flood_layers.append(weights)
        self.flood_indices.append(0)

    # Add layer weights to weight FIFO buffer
    def add_weights(self, weights):
        self.weight_buffer.put(weights)

    # Remove weights from weight buffer given dimensions
    def get_weights(self):
        return self.weight_buffer.get()

    def create_flooded_weights(self):
        completed_layer = False
        for weights, idx in zip(self.flood_layers, self.flood_indices):
            if idx >= weights.shape[0] + weights.shape[1]:
                completed_layer = True
            
            for i in range(self.mmu.shape[0]):
                for j in range(self.mmu.shape[0]):
                    if i < weights.shape[0] and j < weights.shape[1] and i+j <= idx:
                        self.flood_weights[i, j] = weights[i, j]

        if completed_layer:
            self.flood_layers.pop(0)
            self.flood_indices.pop(0)

    def cycle(self):
        if len(self.inputs) != 0:
            if self.current_input_size == 0: # start flooding new weights
                input_shape = self.inputs.pop()
                self.current_input_size = input_shape[1] - 1
                weights = self.weight_buffer.get()
                self.flood_layers.append(weights)
                self.flood_indices.append(0)

                print(self.wait_cycles)
            else:
                self.current_input_size -= 1

        self.create_flooded_weights()
        self.mmu.update_weights(self.flood_weights)
        self.flood_indices = [i+1 for i in self.flood_indices]

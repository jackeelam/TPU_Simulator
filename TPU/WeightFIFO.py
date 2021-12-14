import numpy as np
from queue import Queue

class WeightFIFO:
    """
    TPU Weight FIFO Fetcher

    Stores all the weights and performs the diagonal pipelining of the weights 
    through the systolic array MMU
    """
    def __init__(self, mmu, inputs_data):
        """
        Initialize the Weight FIFO and weight flooding parameters. Flooding means to distribute 
        weights diagonally every cycle, with multiple weight layers allowed to be overlapped in the 
        weight flood at once, moving diagonally each cycle.

        Parameters:
            mmu: Reference to the MMU systolic array
            inputs_data: Reference to the shapes of the inputs for coordination
        """
        self.inputs = inputs_data
        self.weight_buffer = Queue()

        self.mmu = mmu

        self.current_input_size = 0
        self.flood_weights = np.zeros(self.mmu.shape)
        self.flood_layers = []
        self.flood_indices = []

    def initialize_flood(self):
        """
        Pop the first layer weight matrix to begin flooding the MMU weights 
        """
        weights = self.weight_buffer.get()
        self.flood_layers.append(weights)
        self.flood_indices.append(0)

    def add_weights(self, weights):
        """
        Add layer weights to the weight FIFO
        """
        self.weight_buffer.put(weights)

    def create_flooded_weights(self):
        """
        Flood the layer weights throughout the weight matrix based on the current counter
        for each layer
        """
        completed_layer = False
        for weights, idx in zip(self.flood_layers, self.flood_indices):
            if idx >= weights.shape[0] + weights.shape[1]: # check if layer has been fully flooded
                completed_layer = True
            
            for i in range(self.mmu.shape[0]):
                for j in range(self.mmu.shape[0]):
                    # flood the layer up until the counter
                    if i < weights.shape[0] and j < weights.shape[1] and i+j <= idx: 
                        self.flood_weights[i, j] = weights[i, j]

        if completed_layer:
            self.flood_layers.pop(0)
            self.flood_indices.pop(0)

    def cycle(self):
        """
        Cycle the weight fetcher to load in the next layer weights and update the 
        MMU weights after calculating the weight flood matrix
        """
        if len(self.inputs) != 0:
            if self.current_input_size == 0: # start flooding new weights
                input_shape = self.inputs.pop()
                self.current_input_size = input_shape[1] - 1
                weights = self.weight_buffer.get()
                self.flood_layers.append(weights)
                self.flood_indices.append(0)
            else:
                self.current_input_size -= 1

        self.create_flooded_weights()
        self.mmu.update_weights(self.flood_weights)
        self.flood_indices = [i+1 for i in self.flood_indices]
        print(self.flood_weights)
        print()
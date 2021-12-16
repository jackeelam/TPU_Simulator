import numpy as np
from queue import Queue
from dataclasses import dataclass

class MAC:
    """
    TPU Multiply Accumulator

    Performs all systolic array multiplication
    """
    def __init__(self, x, y, mac_left, mac_up, input_buffer, weight_buffer):
        """
        Initialize the MAC with array index and references to neighboring elements
        to read inputs from

        Parameters:
            x: column index
            y: row index
            mac_left: reference to left MAC
            mac_up: reference to upper MAC
            input_buffer: reference to the systolic array buffer from the Unified Buffer
            weight_buffer: 
        """
        self.x = x
        self.y = y

        if x == 0:  # first column
            self.activation_source = input_buffer[y]
        else:
            self.activation_source = mac_left

        if y == 0:  # first row
            self.weight_source = weight_buffer[x]
        else:
            self.weight_source = mac_up

        self.input_weight = None
        self.input_activation = None

        self.result_partial_sum = 0
        self.result_output = 0
        self.result_weight = 0
        self.result_activation = 0

    def read(self):
        """
        Read inputs from left and upper MACs (or from input buffer if the first column)
        """
        if self.x == 0:  # first column
            if self.activation_source.empty():
                self.input_activation = 0
            else:
                self.input_activation = self.activation_source.get()
        else:
            self.input_activation = self.activation_source.result_activation

        if self.y == 0:  # first row
            self.input_result = 0
            if self.weight_source.empty():
                self.input_weight = 0
            else:
                self.input_weight = self.weight_source.get()
        else:
            self.input_weight = self.weight_source.result_weight
            self.input_result = self.weight_source.result_output

    def compute(self):
        """
        Perform systolic array computation to result in output activation and partial sum
        """
        self.result_activation = self.input_activation
        if self.input_activation == 0:
            self.result_weight = self.input_weight
            self.result_output = self.result_partial_sum
            self.result_partial_sum = self.input_result
        else:
            self.result_partial_sum = self.result_partial_sum + (
                self.input_weight * self.input_activation
            )
            self.result_weight = self.input_weight
            self.result_output = 0


class MMU:
    """
    TPU Matrix Multiply Unit

    Systolic Array architecture for matrix multiplication
    """
    def __init__(self, rows, cols, unified_buffer, weight_fetcher, accumulator):
        """
        Initializes the MMU with references to input and output elements

        Paramaters:
            rows: number of rows in the MMU
            cols: number of columns in the MMU
            unified_buffer: reference to the Unified Buffer element for inputs
            weight_fetcher:
            accumulator: reference to Accumulator element for outputs
        """
        self.accumulator = accumulator
        self.array = np.ndarray((rows, cols), dtype=MAC) # define MAC array

        self.unified_buffer = unified_buffer
        self.weight_fetcher = weight_fetcher
        self.shape = [rows, cols]

        # initialize MAC array
        for i in range(rows):
            for j in range(cols):
                mac_up = self.array[i - 1, j] if i != 0 else None
                mac_left = self.array[i, j - 1] if j != 0 else None
                self.array[i, j] = MAC(j, i, mac_left, mac_up, unified_buffer.systolic_array_buffer, weight_fetcher.weight_buffer)

    def cycle(self):
        """
        Cycle the MMU to have all MACs read inputs and perform computations
        """
        for row in self.array:
            for mac in row:
                mac.read()
        for row in self.array:
            for mac in row:
                mac.compute()
        
        # store final row in the accumulators
        self.accumulator.accumulate_partial_sum(self.array[-1])

    def display(self):
        print(np.array([[mac.result_output for mac in row] for row in self.array]))

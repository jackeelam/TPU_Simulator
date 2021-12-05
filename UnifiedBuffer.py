import numpy as np
from queue import Queue

# Unified Buffer


class UnifiedBuffer:
    def __init__(self, input, systolic_array_size=256, sram_length=5000):
        self.sram = [[]] * sram_length
        self.systolic_array_size = systolic_array_size
        self.systolic_array_buffer = self.systolic_array_setup(input.T)

    # Given 2D input, reformat so that it is triangle padded with zeros
    def systolic_array_setup(self, input):
        # Initialize input_buffer of size of original input
        buffer = []
        for row in range(self.systolic_array_size):
            buffer.append(Queue())
            if row >= len(input):
                continue

            # pad with traingle of zeroes
            for i in range(row):
                buffer[row].put(0)

            # insert input
            for element in input[row]:
                buffer[row].put(element)

        return buffer

    def store_accumulator_results(self, acc, rows):
        
        pass

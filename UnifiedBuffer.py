import numpy as np
from queue import Queue

# Unified Buffer


class UnifiedBuffer:
    def __init__(self, systolic_array_size=256, sram_length=5000):
        self.sram = []
        self.systolic_array_size = systolic_array_size

    def get_systolic_array_buffer(self, sram_index=0):
        input = self.sram.pop(sram_index)
        return self.systolic_array_setup(input.T)

    # Given 2D input, reformat so that it is triangle padded with zeros
    def systolic_array_setup(self, input):
        # Initialize buffer of size of original input
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

    def store_acc(self, accumulator, rows=None):
        n_rows = max(accumulator.index) if rows == None else rows
        mat = accumulator.acc[:n_rows, :accumulator.width+1].copy()
        self.sram.append(mat)
        accumulator.acc[:n_rows, :accumulator.width+1] = 0

    def store_input(self, input):
        self.sram.append(input)

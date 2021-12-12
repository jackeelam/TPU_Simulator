import numpy as np
from queue import Queue

# Unified Buffer


class UnifiedBuffer:
    def __init__(self, systolic_array_size):
        self.sram = []
        self.sram_outputs = []
        self.systolic_array_size = systolic_array_size

        self.systolic_array_buffer = []
        for i in range(systolic_array_size):
            self.systolic_array_buffer.append(Queue())

    def update_systolic_array_buffer(self, sram_index=0):
        input = self.sram.pop(sram_index).T
        for row in range(self.systolic_array_size):
            # pad with traingle of zeroes
            for i in range(row - self.systolic_array_buffer[row].qsize()):
                self.systolic_array_buffer[row].put(0)

            # insert input
            for element in input[row]:
                self.systolic_array_buffer[row].put(element)
    
    def store_acc(self, accumulator, rows=None):
        n_rows = max(accumulator.index) if rows == None else rows
        mat = accumulator.get_block(n_rows)
        self.sram_outputs.append(mat)

    def store_input(self, input):
        self.sram.append(input)
        # self.update_systolic_array_buffer()

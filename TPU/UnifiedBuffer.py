import numpy as np
from queue import Queue

# Unified Buffer


class UnifiedBuffer:
    def __init__(self, systolic_array_rows):
        self.sram = []
        self.sram_outputs = []
        self.systolic_array_size = systolic_array_rows

        self.systolic_array_buffer = []
        for i in range(systolic_array_rows):
            self.systolic_array_buffer.append(Queue())

    def update_systolic_array_buffer(self, offset=0, sram_index=0):
        # input = self.sram.pop(sram_index).T
        input = self.sram[-1].T
        for row in range(self.systolic_array_size):
            # pad with traingle of zeroes
            for i in range(np.clip(row - self.systolic_array_buffer[row].qsize(), 0, None) + offset):
                self.systolic_array_buffer[row].put(0)

            # insert input
            buffer = np.zeros(input[0].shape[0])
            if row < len(input):
                buffer = input[row]
            for element in buffer:
                self.systolic_array_buffer[row].put(element)

    def display_systolic_array_buffer(self):
        for row in self.systolic_array_buffer:
            print(',\t'.join([str(i) for i in row.queue]))

    def allocate_output(self, shape):
        self.sram_outputs.append(np.zeros(shape))

    def store_acc(self, accumulator, shape, index=0, start_row=0, start_col=0):
        mat = accumulator.get_block(shape)
        self.sram_outputs[index][start_row:start_row+mat.shape[0], start_col:start_col+mat.shape[1]] = mat

    def store_input(self, input):
        offset = 0 if len(self.sram) == 0 else np.clip(input.shape[0] - self.sram[-1].shape[0], 0, None)
        self.sram.append(input)
        self.update_systolic_array_buffer(offset)

import numpy as np
from queue import Queue

# Unified Buffer


class UnifiedBuffer:
    def __init__(self, input):
        self.input_rows = len(input)
        self.input_buffer = self.systolic_array_setup(input, self.input_rows)

    # Given 2D input, reformat so that it is triangle padded with zeros
    def systolic_array_setup(self, input, input_rows):

        # Initialize input_buffer of size of original input
        input_buffer = []
        for row in range(input_rows):
            input_buffer.append(Queue())

            # pad with traingle of zeroes
            for i in range(row):
                input_buffer[row].put(0)

            # insert input
            for element in input[row]:
                input_buffer[row].put(element)

        return input_buffer

    # Add input to the activation FIFO buffer
    def append_input(self, additional_input):
        for row in range(self.input_rows):

            # insert input
            for element in additional_input[row]:
                self.input_buffer[row].put(element)

    # Remove a column from the queue to process the input
    def get_next_col_input(self):
        col = []
        for queue in self.input_buffer:
            col.append(queue.get())

        return col

    def store_accumulator_results(self, acc):
        pass

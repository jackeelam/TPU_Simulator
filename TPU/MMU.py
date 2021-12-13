import numpy as np
from queue import Queue
from dataclasses import dataclass

class MAC:
    def __init__(self, x, y, mac_left, mac_up, input_buffer):
        self.x = x
        self.y = y

        self.current_op_id = 0

        if x == 0:  # first column
            self.activation_source = input_buffer[y]
        else:
            self.activation_source = mac_left

        if y == 0:  # first row
            self.partial_sum_source = None
        else:
            self.partial_sum_source = mac_up

        self.weight = 0

        self.input_partial_sum = None
        self.input_activation = None

        self.result_partial_sum = 0
        self.result_activation = 0

    def read(self):
        if self.x == 0:  # first column
            if self.activation_source.empty():
                self.input_activation = 0
            else:
                self.input_activation = self.activation_source.get()
        else:
            self.input_activation = self.activation_source.result_activation

        if self.input_activation == None:
            print("wtf")

        if self.y == 0:  # first row
            self.input_partial_sum = 0
        else:
            self.input_partial_sum = self.partial_sum_source.result_partial_sum

    def compute(self):
        self.result_partial_sum = self.input_partial_sum + (
            self.weight * self.input_activation
        )
        self.result_activation = self.input_activation

class MMU:
    def __init__(self, rows, cols, unified_buffer, accumulator):
        self.accumulator = accumulator
        self.array = np.ndarray((rows, cols), dtype=MAC)

        self.unified_buffer = unified_buffer
        # self.weight_fifo = weight_fifo

        self.shape = [rows, cols]

        # self.current_weights = self.weight_fifo.get_weights()

        for i in range(rows):
            for j in range(cols):
                mac_up = self.array[i - 1, j] if i != 0 else None
                mac_left = self.array[i, j - 1] if j != 0 else None
                self.array[i, j] = MAC(j, i, mac_left, mac_up, self.unified_buffer.systolic_array_buffer)

    def update_weights(self, weights):
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                self.array[i, j].weight = weights[i, j]

    def set_weight(self, row, col, weight):
        self.array[row, col] = weight

    def cycle(self, input_size=(3,3)):
        N = len(input_size)
        for row in self.array:
            for mac in row:
                mac.read()
        for row in self.array:
            for mac in row:
                mac.compute()
        
        self.accumulator.accumulate_partial_sum(self.array[-1])


@dataclass
class MMUInputType:
    input_id: int
    value: int
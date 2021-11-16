import numpy as np


class MAC:
    def __init__(self, x, y, mac_left, mac_up, input_buffer):
        self.x = x
        self.y = y

        if x == 0:  # first column
            self.activation_source = input_buffer[y]
        else:
            self.activation_source = mac_left

        if y == 0:  # first row
            self.partial_sum_source = None
        else:
            self.partial_sum_source = mac_up

        self.weight = None

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
    def __init__(self, rows, cols, mmu_inputs, weights, accumulator):
        self.accumulator = accumulator
        self.array = np.ndarray((rows, cols), dtype=MAC)
        for i in range(rows):
            for j in range(cols):
                mac_up = self.array[i - 1, j] if i != 0 else None
                mac_left = self.array[i, j - 1] if j != 0 else None
                self.array[i, j] = MAC(j, i, mac_left, mac_up, mmu_inputs)

        for i in range(len(weights)):
            for j in range(len(weights[0])):
                self.array[i, j].weight = weights[i, j]

    def cycle(self):
        for row in self.array:
            for mac in row:
                mac.read()

        for row in self.array:
            for mac in row:
                mac.compute()

        self.accumulator.add_partial_sum(self.array[-1])

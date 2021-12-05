import numpy as np


class Accumulator:
    def __init__(self, n_accumulators, acc_size):
        self.acc = np.zeros((acc_size, n_accumulators))
        self.index = [0] * n_accumulators
        self.width = 0

    def add_partial_sum(self, macs):
        buffer = np.array([mac.result_partial_sum for mac in macs])
        if np.all(buffer == 0):
            return
        
        curr_width = 0
        for i in range(len(buffer)):
            if buffer[i] == 0:
                self.width = max(self.width, curr_width)
                continue
            self.acc[self.index[i], i] = buffer[i]
            self.index[i] += 1
            curr_width = i + 1

    def display(self):
        print('ACCUMULATOR: ')
        for i in range(max(self.index)):
            print('\t'.join(str(val) for val in self.acc[i, :self.width+1]))
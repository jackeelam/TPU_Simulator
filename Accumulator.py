import numpy as np


class Accumulator:
    def __init__(self, n_accumulators, acc_size):
        self.acc = np.zeros((acc_size, n_accumulators))
        self.acc_idx = 0
        self.width = 0

    def add_partial_sum(self, macs):
        self.width = len(macs)
        self.acc[self.acc_idx, :len(macs)] = [mac.result_partial_sum for mac in macs]
        self.acc_idx += 1

    def display(self):
        print('ACCUMULATOR: ')
        for i in range(self.acc_idx):
            print('\t'.join(str(val) for val in self.acc[i, :self.width]))
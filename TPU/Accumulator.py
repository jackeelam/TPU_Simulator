import numpy as np


class Accumulator:
    def __init__(self, n_accumulators, acc_size):
        self.acc = np.zeros((acc_size, n_accumulators), dtype=np.int32)
        self.index = np.zeros(n_accumulators, dtype=int)
        self.acc_cap = acc_size

    def accumulate_partial_sum(self, macs):
        buffer = np.array([mac.result_partial_sum for mac in macs])
        if np.all(buffer == 0):
            return
        
        for i in range(len(buffer)):
            if buffer[i] == 0:
                continue
            self.acc[self.index[i], i] += buffer[i]
            self.index[i] = (self.index[i] + 1) % self.acc_cap
    
    def reset_acc_cap(self):
        self.acc_cap = len(self.acc)

    def display(self, n=10):
        print('ACCUMULATOR: ')
        for i in range(n):
            print('\t'.join(str(val) for val in self.acc[i]))

    def get_block(self, shape):
        mat = self.acc[:shape[0], :shape[1]].copy()
        self.acc[:shape[0], :shape[1]] = 0
        self.acc = np.roll(self.acc, -shape[0], axis=0)
        self.index[:shape[1]] -= shape[0]
        return mat
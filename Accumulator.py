import numpy as np


class Accumulator:
    def __init__(self, n_accumulators, acc_size):
        self.acc = np.zeros((acc_size, n_accumulators))
        self.acc_idx = 0

    def add_partial_sum(self, psum):
        self.acc[self.acc_idx, :] = psum
        self.acc_idx += 1

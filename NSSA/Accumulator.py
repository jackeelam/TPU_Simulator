import numpy as np


class Accumulator:
    """
    TPU Accumulator

    The accumulator aligns and stores the partial sum of the last row of the systolic array MMU.
    It can also perform addition operations by resetting the 'cap' for each accumulator, which
    resets the index to start overlapping and adding successive values. 
    """
    def __init__(self, n_accumulators, acc_size):
        """
        Initialize all accumulators

        Parameters:
            n_accumulators: Number of accumulators, equal to number of columns of the systolic array
            acc_size: The length of each accumulator
        """
        # array of accumulators
        self.acc = np.zeros((acc_size, n_accumulators), dtype=np.int32) 
        self.index = np.zeros(n_accumulators, dtype=int) # current index to store values at for each accumulator
        self.cap = np.full(n_accumulators, acc_size)
        self.acc_size = acc_size

    def accumulate_partial_sum(self, macs):
        """
        Stores the partial sums of the last MMU row in the accumulators.
        """
        buffer = np.array([mac.result_output for mac in macs])
        if np.all(buffer == 0): # buffer of 0's means no relevent output
            return
        
        for i in range(len(buffer)):
            if buffer[i] == 0: # do not store 0's
                continue
            self.acc[self.index[i], i] += buffer[i] # add the value to the accumulator
            self.index[i] = self.index[i] + 1 # increment current accumulator index for the next value
            if self.index[i] == self.cap[i]: # reset cap
                self.index[i] = 0
                self.cap[i] = self.acc_size
    
    def set_acc_cap(self, cap):
        """
        Sets the cap for the accumulators to start overlapping successive results
        """
        self.cap = np.full_like(self.cap, cap)

    def display(self, n=20):
        """
        Test function to print the top n rows of the accumulators
        """
        print('ACCUMULATOR: ')
        for i in range(n):
            print('\t'.join(str(val) for val in self.acc[i]))

    def get_block(self, shape):
        """
        Extract a region of the accumulators with the provided shape and shift all successive
        columns to the top
        """
        mat = self.acc[:shape[0], :shape[1]].copy()
        self.acc[:shape[0], :shape[1]] = 0
        self.acc = np.roll(self.acc, -shape[0], axis=0)
        self.index[:shape[1]] -= shape[0]
        return mat
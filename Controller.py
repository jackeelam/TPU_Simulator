import numpy as np
from MMU import MMU
from Accumulator import Accumulator
from queue import Queue

# Sample input and weights
input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

weights = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])


def systolic_array_setup(input, input_rows):
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

        for i in range(input_rows - row - 1):
            input_buffer[row].put(0)

    return input_buffer


ground_truth = np.matmul(input, weights)

sa_input = systolic_array_setup(input.T, len(input))
mmu = MMU(3, 3, sa_input, weights)
acc = Accumulator(4096, 256)

cycles = 7

for i in range(cycles):
    mmu.cycle()
    acc.add_partial_sum(mmu.array[-1])


print()
print("Ground truth: ")
print(ground_truth)

import numpy as np
from queue import Queue

from UnifiedBuffer import UnifiedBuffer
from MMU import MMU
from Accumulator import Accumulator
from WeightFIFO import WeightFIFO

# Sample input and weights
inputMatrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

weights = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

ub = UnifiedBuffer(inputMatrix)
wf = WeightFIFO(weights)

acc = Accumulator(3, 256)
mmu = MMU(3, 3, ub.systolic_array_buffer, weights, acc)

cycles = 7

for i in range(cycles):
    mmu.cycle()
    
acc.display()


ground_truth = np.matmul(inputMatrix, weights)
print()
print("MATRIX MULTIPLICATION: ")
print(ground_truth)

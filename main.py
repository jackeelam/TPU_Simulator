import numpy as np
from queue import Queue

from UnifiedBuffer import UnifiedBuffer
from MMU import MMU
from Accumulator import Accumulator
from WeightFIFO import WeightFIFO

# Sample input and weights
inputMatrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

weights = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

ub = UnifiedBuffer()
ub.store_input(inputMatrix)
wf = WeightFIFO(weights)

# print(wf.weight_buffer[0].queue)
acc = Accumulator(3, 256)
mmu = MMU(3, 3, ub.get_systolic_array_buffer(), wf.weight_buffer, acc)

cycles = 7

for i in range(cycles):
    # print("Cycle {}".format(i))
    mmu.cycle()

#Print out the MMU
print("MMU")
for i in range(3):
    r = ""
    for j in range(3):
        r += str(mmu.array[i][j].result_partial_sum) + " "
    print(r)
    

# acc.display(), already aligned, just take from the array of MACs
# ub.store_acc(acc)

ground_truth = np.matmul(inputMatrix, weights)
print()
print("MATRIX MULTIPLICATION: ")
print(ground_truth)

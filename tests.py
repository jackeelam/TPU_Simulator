import numpy as np

from utilities import tile_matrix
from TPU import *

def test_single_input():
    # Sample input and weights
    inputMatrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    weights = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    ub = UnifiedBuffer(3)
    ub.store_input(inputMatrix)
    wf = WeightFIFO()
    wf.add_weights(weights)

    acc = Accumulator(3, 256)
    mmu = MMU(3, 3, ub, wf, acc)

    cycles = 7
    mmu.update_weights()
    ub.update_systolic_array_buffer()
    for i in range(cycles):
        mmu.cycle()

    ub.store_acc(acc)

    ground_truth = np.matmul(inputMatrix, weights)
    print("SYSTOLIC ARRAY MULTIPLICATION: ")
    print(ub.sram_outputs[-1])
    print()
    print("MATRIX MULTIPLICATION: ")
    print(ground_truth)

def test_double_input_same_weights():
    # Sample input and weights
    inputMatrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inputMatrix2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 1]])
    weights = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    ub = UnifiedBuffer(3)
    ub.store_input(inputMatrix1)
    ub.store_input(inputMatrix2)
    wf = WeightFIFO()
    wf.add_weights(weights)

    acc = Accumulator(3, 256)
    mmu = MMU(3, 3, ub, wf, acc)

    cycles1 = 7
    cycles2 = 7

    mmu.update_weights()
    ub.update_systolic_array_buffer()
    for i in range(cycles1):
        mmu.cycle()
    ub.store_acc(acc, rows=len(inputMatrix1))
    ub.update_systolic_array_buffer()
    for i in range(cycles2):
        mmu.cycle()
    ub.store_acc(acc, rows=len(inputMatrix2))

    ground_truth1 = np.matmul(inputMatrix1, weights)
    print("SYSTOLIC ARRAY MULTIPLICATION 1: ")
    print(ub.sram_outputs[-2])
    print()
    print("MATRIX MULTIPLICATION 1: ")
    print(ground_truth1)
    print()

    ground_truth2 = np.matmul(inputMatrix2, weights)
    print("SYSTOLIC ARRAY MULTIPLICATION 2: ")
    print(ub.sram_outputs[-1])
    print()
    print("MATRIX MULTIPLICATION 2: ")
    print(ground_truth2)

def test_large_single_input(size=(6,6), largest_block_size=3):
    inputMatrix = np.random.randint(0, 50, size)
    weights = np.random.randint(0, 50, size)

    A, B, C, D = tile_matrix(inputMatrix, largest_block_size)
    E, F, G, H = tile_matrix(weights, largest_block_size)

    ub = UnifiedBuffer(3)
    ub.store_input(A)
    ub.store_input(B)
    wf = WeightFIFO()
    wf.add_weights(E)
    wf.add_weights(G)

    acc = Accumulator(3, 256)
    mmu = MMU(3, 3, ub, wf, acc)

    cycles1 = 7
    cycles2 = 7

    acc.acc_cap = len(A)
    mmu.update_weights()
    ub.update_systolic_array_buffer()
    for i in range(cycles1):
        mmu.cycle()
    acc.display()
    mmu.update_weights()
    ub.update_systolic_array_buffer()
    for i in range(cycles2):
        mmu.cycle()
    acc.display()

    ub.store_acc(acc)

    ground_truth = np.matmul(A, E) + np.matmul(B, G)
    print()
    print("MATRIX MULTIPLICATION: ")
    print(ground_truth)
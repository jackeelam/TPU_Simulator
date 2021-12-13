import numpy as np

from utilities import tile_matrix
from TPU import *

def test_single_input(mmu_rows=3, mmu_cols=3):
    # Sample input and weights
    inputMatrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    weights = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    input_shapes = [inputMatrix.shape]

    ub = UnifiedBuffer(mmu_rows)
    ub.store_input(inputMatrix)

    acc = Accumulator(mmu_cols, 256)
    mmu = MMU(mmu_rows, mmu_cols, ub, acc)
    wf = WeightFIFO(mmu, input_shapes)

    cycles = 7

    for i in range(cycles):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc)

    ground_truth = np.matmul(inputMatrix, weights)
    print("SYSTOLIC ARRAY MULTIPLICATION: ")
    print(ub.sram_outputs[-1])
    print()
    print("MATRIX MULTIPLICATION: ")
    print(ground_truth)

def test_double_input_same_weights(mmu_rows=3, mmu_cols=3):
    # Sample input and weights
    inputMatrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inputMatrix2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 1]])
    weights = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    input_shapes = [inputMatrix1.shape, inputMatrix2.shape]

    ub = UnifiedBuffer(mmu_rows)
    ub.store_input(inputMatrix1)
    ub.store_input(inputMatrix2)

    acc = Accumulator(mmu_cols, 256)
    mmu = MMU(mmu_rows, mmu_cols, ub, acc)
    wf = WeightFIFO(mmu, input_shapes)
    wf.add_weights(weights)
    wf.add_weights(weights)

    cycles1 = 7
    cycles2 = 3

    for i in range(cycles1):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, rows=len(inputMatrix1))
    for i in range(cycles2):
        wf.cycle()
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


def test_double_input_different_weights(mmu_rows=3, mmu_cols=3):
    # Sample input and weights
    inputMatrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inputMatrix2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 1]])
    weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    weights2 = np.array([[40, 50, 90], [70, 80, 30], [10, 20, 60]])
    input_shapes = [inputMatrix1.shape, inputMatrix2.shape]

    ub = UnifiedBuffer(mmu_rows)
    ub.store_input(inputMatrix1)
    ub.store_input(inputMatrix2)

    acc = Accumulator(mmu_cols, 256)
    mmu = MMU(mmu_rows, mmu_cols, ub, acc)
    wf = WeightFIFO(mmu, input_shapes)
    wf.add_weights(weights1)
    wf.add_weights(weights2)

    cycles1 = 7
    cycles2 = 3

    for i in range(cycles1):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, rows=len(inputMatrix1))
    for i in range(cycles2):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, rows=len(inputMatrix2))

    ground_truth1 = np.matmul(inputMatrix1, weights1)
    print("SYSTOLIC ARRAY MULTIPLICATION 1: ")
    print(ub.sram_outputs[-2])
    print()
    print("MATRIX MULTIPLICATION 1: ")
    print(ground_truth1)
    print()

    ground_truth2 = np.matmul(inputMatrix2, weights2)
    print("SYSTOLIC ARRAY MULTIPLICATION 2: ")
    print(ub.sram_outputs[-1])
    print()
    print("MATRIX MULTIPLICATION 2: ")
    print(ground_truth2)

def test_large_single_input(mmu_rows=3, mmu_cols=3, input_size=(6,6), largest_block_size=3):
    inputMatrix = np.random.randint(1, 50, input_size)
    weights = np.random.randint(1, 50, input_size)

    A, B, C, D = tile_matrix(inputMatrix, largest_block_size)
    E, F, G, H = tile_matrix(weights, largest_block_size)

    input_shapes = [A.shape, B.shape]

    ub = UnifiedBuffer(mmu_rows)
    ub.store_input(A)
    ub.store_input(B)

    acc = Accumulator(mmu_cols, 256)
    mmu = MMU(mmu_rows, mmu_cols, ub, acc)
    wf = WeightFIFO(mmu, input_shapes)
    wf.add_weights(E)
    wf.add_weights(G)

    cycles1 = 7
    cycles2 = 3

    # acc.acc_cap = len(A)
    for i in range(cycles1):
        mmu.cycle()
    # ub.store_acc(acc, rows=len(A))
    acc.display()
    for i in range(cycles2):
        mmu.cycle()
    # ub.store_acc(acc, rows=len(B))
    acc.display()

    ground_truth = np.matmul(A, E) + np.matmul(B, G)
    print()
    print("MATRIX MULTIPLICATION: ")
    print(ground_truth)
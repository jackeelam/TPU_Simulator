import numpy as np

from utilities import tile_matrix, calculate_num_cycles_TPU
from TPU import *

MMU_ROWS = 100
MMU_COLS = 100

ACCUMULATOR_SIZE = 4096


def createTPU(inputs):
    ub = UnifiedBuffer(MMU_ROWS)
    acc = Accumulator(MMU_COLS, ACCUMULATOR_SIZE)
    mmu = MMU(MMU_ROWS, MMU_COLS, ub, acc)
    input_shapes = [input.shape for input in inputs]
    wf = WeightFIFO(mmu, input_shapes)

    return ub, acc, mmu, wf

def test_single_input():
    inputMatrix = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
    weights = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))
    output_shape = [inputMatrix.shape[0], weights.shape[1]]

    ub, acc, mmu, wf = createTPU([inputMatrix])

    ub.store_input(inputMatrix)
    wf.add_weights(weights)

    cycles = calculate_num_cycles_TPU(inputMatrix.shape, output_shape, MMU_ROWS)

    ub.allocate_output(output_shape)
    for i in range(cycles):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shape)

    ground_truth = np.matmul(inputMatrix, weights)
    result = ub.sram_outputs[0]

    assert np.array_equal(result, ground_truth)

def test_single_input_small():
    inputMatrix = np.random.randint(1, 10, (MMU_ROWS//2, MMU_COLS//2))
    weights = np.random.randint(1, 100, (MMU_ROWS//2, MMU_COLS//2))
    output_shape = [inputMatrix.shape[0], weights.shape[1]]

    ub, acc, mmu, wf = createTPU([inputMatrix])

    ub.store_input(inputMatrix)
    wf.add_weights(weights)

    cycles = calculate_num_cycles_TPU(inputMatrix.shape, output_shape, MMU_ROWS)

    ub.allocate_output(output_shape)
    for i in range(cycles):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shape)

    ground_truth = np.matmul(inputMatrix, weights)
    result = ub.sram_outputs[0]
    assert np.array_equal(result, ground_truth)

def test_single_input_rectangular_horizontal():
    inputMatrix = np.random.randint(1, 10, (MMU_ROWS//3, MMU_COLS))
    weights = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS//2))
    output_shape = [inputMatrix.shape[0], weights.shape[1]]

    ub, acc, mmu, wf = createTPU([inputMatrix])

    ub.store_input(inputMatrix)
    wf.add_weights(weights)

    cycles = calculate_num_cycles_TPU(inputMatrix.shape, output_shape, MMU_ROWS)

    ub.allocate_output(output_shape)
    for i in range(cycles):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shape)

    ground_truth = np.matmul(inputMatrix, weights)
    result = ub.sram_outputs[0]
    assert np.array_equal(result, ground_truth)


def test_single_input_rectangular_vertical():
    inputMatrix = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS//3))
    weights = np.random.randint(1, 100, (MMU_ROWS//3, MMU_COLS//2))
    output_shape = [inputMatrix.shape[0], weights.shape[1]]

    ub, acc, mmu, wf = createTPU([inputMatrix])

    ub.store_input(inputMatrix)
    wf.add_weights(weights)
    
    cycles = calculate_num_cycles_TPU(inputMatrix.shape, output_shape, MMU_ROWS)

    ub.allocate_output(output_shape)
    for i in range(cycles):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shape)

    ground_truth = np.matmul(inputMatrix, weights)
    result = ub.sram_outputs[0]
    assert np.array_equal(result, ground_truth)

def test_double_input_same_weights():
    inputMatrix1 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
    inputMatrix2 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
    weights = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))
    output_shapes = [[inputMatrix1.shape[0], weights.shape[1]], [inputMatrix2.shape[0], weights.shape[1]]]

    ub, acc, mmu, wf = createTPU([inputMatrix1, inputMatrix2])

    ub.store_input(inputMatrix1)
    ub.store_input(inputMatrix2)

    wf.add_weights(weights)
    wf.add_weights(weights)

    cycles1 = calculate_num_cycles_TPU(inputMatrix1.shape, output_shapes[0], MMU_ROWS)
    cycles2 = calculate_num_cycles_TPU(inputMatrix2.shape, output_shapes[1], MMU_ROWS, inputMatrix1.shape)

    ub.allocate_output(output_shapes[0])
    ub.allocate_output(output_shapes[1])
    for i in range(cycles1):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[0], index=0)
    for i in range(cycles2):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[1], index=1)

    ground_truth1 = np.matmul(inputMatrix1, weights)
    result1 = ub.sram_outputs[0]
    assert np.array_equal(ground_truth1, result1)

    ground_truth2 = np.matmul(inputMatrix2, weights)
    result2 = ub.sram_outputs[1]
    assert np.array_equal(ground_truth2, result2)


def test_double_input_different_weights():
    inputMatrix1 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
    inputMatrix2 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
    weights1 = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))
    weights2 = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))
    
    output_shapes = [[inputMatrix1.shape[0], weights1.shape[1]], [inputMatrix2.shape[0], weights2.shape[1]]]

    ub, acc, mmu, wf = createTPU([inputMatrix1, inputMatrix2])

    ub.store_input(inputMatrix1)
    ub.store_input(inputMatrix2)

    wf.add_weights(weights1)
    wf.add_weights(weights2)

    cycles1 = calculate_num_cycles_TPU(inputMatrix1.shape, output_shapes[0], MMU_ROWS)
    cycles2 = calculate_num_cycles_TPU(inputMatrix2.shape, output_shapes[1], MMU_ROWS, inputMatrix1.shape)

    ub.allocate_output(output_shapes[0])
    ub.allocate_output(output_shapes[1])
    for i in range(cycles1):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[0], index=0)
    for i in range(cycles2):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[1], index=1)

    ground_truth1 = np.matmul(inputMatrix1, weights1)
    result1 = ub.sram_outputs[0]
    assert np.array_equal(ground_truth1, result1)

    ground_truth2 = np.matmul(inputMatrix2, weights2)
    result2 = ub.sram_outputs[1]
    assert np.array_equal(ground_truth2, result2)


def test_double_input_different_weight_different_size_larger():
    inputMatrix1 = np.random.randint(1, 10, (MMU_ROWS//2, MMU_COLS//2))
    inputMatrix2 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
    weights1 = np.random.randint(1, 100, (MMU_ROWS//2, MMU_COLS//2))
    weights2 = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))
    
    output_shapes = [[inputMatrix1.shape[0], weights1.shape[1]], [inputMatrix2.shape[0], weights2.shape[1]]]

    ub, acc, mmu, wf = createTPU([inputMatrix1, inputMatrix2])

    ub.store_input(inputMatrix1)
    ub.store_input(inputMatrix2)

    wf.add_weights(weights1)
    wf.add_weights(weights2)

    cycles1 = calculate_num_cycles_TPU(inputMatrix1.shape, output_shapes[0], MMU_ROWS)
    cycles2 = calculate_num_cycles_TPU(inputMatrix2.shape, output_shapes[1], MMU_ROWS, inputMatrix1.shape)
    
    ub.allocate_output(output_shapes[0])
    ub.allocate_output(output_shapes[1])
    for i in range(cycles1):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[0], index=0)
    for i in range(cycles2):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[1], index=1)

    ground_truth1 = np.matmul(inputMatrix1, weights1)
    result1 = ub.sram_outputs[0]
    assert np.array_equal(ground_truth1, result1)

    ground_truth2 = np.matmul(inputMatrix2, weights2)
    result2 = ub.sram_outputs[1]
    assert np.array_equal(ground_truth2, result2)

def test_double_input_different_weight_different_size_smaller():
    inputMatrix1 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
    inputMatrix2 = np.random.randint(1, 10, (MMU_ROWS//2, MMU_COLS//2))
    weights1 = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))
    weights2 = np.random.randint(1, 100, (MMU_ROWS//2, MMU_COLS//2))
    
    output_shapes = [[inputMatrix1.shape[0], weights1.shape[1]], [inputMatrix2.shape[0], weights2.shape[1]]]

    ub, acc, mmu, wf = createTPU([inputMatrix1, inputMatrix2])

    ub.store_input(inputMatrix1)
    ub.store_input(inputMatrix2)

    wf.add_weights(weights1)
    wf.add_weights(weights2)

    cycles1 = calculate_num_cycles_TPU(inputMatrix1.shape, output_shapes[0], MMU_ROWS)
    cycles2 = calculate_num_cycles_TPU(inputMatrix2.shape, output_shapes[1], MMU_ROWS, inputMatrix1.shape)

    ub.allocate_output(output_shapes[0])
    ub.allocate_output(output_shapes[1])
    for i in range(cycles1):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[0], index=0)
    for i in range(cycles2):
        wf.cycle()
        mmu.cycle()
    ub.store_acc(acc, output_shapes[1], index=1)

    ground_truth1 = np.matmul(inputMatrix1, weights1)
    result1 = ub.sram_outputs[0]
    assert np.array_equal(ground_truth1, result1)

    ground_truth2 = np.matmul(inputMatrix2, weights2)
    result2 = ub.sram_outputs[1]
    assert np.array_equal(ground_truth2, result2)

def test_large_single_input():
    inputMatrix = np.random.randint(1, 10, (MMU_ROWS*2, MMU_COLS*2))
    weights = np.random.randint(1, 10, (MMU_ROWS*2, MMU_COLS*2))

    A, B, C, D = tile_matrix(inputMatrix, MMU_ROWS)
    E, F, G, H = tile_matrix(weights, MMU_ROWS)

    input_shapes = [A.shape, B.shape, C.shape, D.shape, A.shape, B.shape, C.shape, D.shape]
    output_shape = [inputMatrix.shape[0], weights.shape[1]]

    sub_output_shapes = [
        [A.shape[0], E.shape[1]],
        [B.shape[0], G.shape[1]],
        [A.shape[0], F.shape[1]],
        [B.shape[0], H.shape[1]],
        [C.shape[0], E.shape[1]],
        [D.shape[0], G.shape[1]],
        [C.shape[0], F.shape[1]],
        [D.shape[0], H.shape[1]],
    ]

    ub, acc, mmu, wf = createTPU([A, B, C, D, A, B, C, D])

    ub.store_input(A)
    ub.store_input(B)
    ub.store_input(A)
    ub.store_input(B)
    ub.store_input(C)
    ub.store_input(D)
    ub.store_input(C)
    ub.store_input(D)

    wf.add_weights(E)
    wf.add_weights(G)
    wf.add_weights(F)
    wf.add_weights(H)
    wf.add_weights(E)
    wf.add_weights(G)
    wf.add_weights(F)
    wf.add_weights(H)

    ub.allocate_output(output_shape)

    for i in range(4): # tiled into 4 subsections
        shape = sub_output_shapes[i*2]
        cycles1 = calculate_num_cycles_TPU(input_shapes[i*2], shape, MMU_ROWS, None if i == 0 else input_shapes[i*2-1])
        cycles2 = calculate_num_cycles_TPU(input_shapes[i*2+1], shape, MMU_ROWS, input_shapes[i*2])

        acc.set_acc_cap(shape[0])
        for c in range(cycles1):
            wf.cycle()
            mmu.cycle()
        acc.set_acc_cap(acc.acc_size)
        for c in range(cycles2):
            wf.cycle()
            mmu.cycle()

        ub.store_acc(acc, shape=shape, start_row=shape[0]*(i//2), start_col=shape[1]*(i%2))


    ground_truth = np.matmul(inputMatrix, weights)
    result = ub.sram_outputs[0]
    assert np.array_equal(result, ground_truth)


if __name__ == '__main__':
    test_single_input()
    test_single_input_small()
    test_single_input_rectangular_horizontal()
    test_single_input_rectangular_vertical()
    test_double_input_same_weights()
    test_double_input_different_weights()
    test_double_input_different_weight_different_size_larger()
    test_double_input_different_weight_different_size_smaller()
    test_large_single_input()
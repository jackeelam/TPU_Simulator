import numpy as np
from utilities import calculate_num_cycles_TPU, calculate_num_cycles_NSSA

import matplotlib.pyplot as plt

MMU_ROWS = 256
MMU_COLS = 256
ACCUMULATOR_SIZE = 4096

def calculate_TPU_cycle_time(n_matrices):
    inputs = np.random.randint(1, 5, (n_matrices, MMU_ROWS, MMU_COLS))
    weights = np.random.randint(1, 5, (n_matrices, MMU_ROWS, MMU_COLS))

    input_shapes = [i.shape for i in inputs]
    output_shapes = [[i.shape[0], w.shape[1]] for i, w in zip(inputs, weights)]

    from TPU import UnifiedBuffer, MMU, Accumulator, WeightFIFO

    ub = UnifiedBuffer(MMU_ROWS)
    acc = Accumulator(MMU_COLS, ACCUMULATOR_SIZE)
    mmu = MMU(MMU_ROWS, MMU_COLS, ub, acc)
    wf = WeightFIFO(mmu, input_shapes.copy())

    total_cycles = 0

    for i in range(n_matrices):
        ub.store_input(inputs[i])
        wf.add_weights(weights[i])

    for i in range(n_matrices):
        print(f'TPU Processing matrix {i+1}/{n_matrices}')
        shape = output_shapes[i]
        cycles = calculate_num_cycles_TPU(input_shapes[i], shape, MMU_ROWS, None if i == 0 else input_shapes[i-1])

        ub.allocate_output(shape)
        for c in range(cycles):
            wf.cycle()
            mmu.cycle()
        ub.store_acc(acc, shape)

        ground_truth = np.matmul(inputs[i], weights[i])
        result = ub.sram_outputs[-1]
        assert np.array_equal(ground_truth, result)
        total_cycles += cycles

    return total_cycles

def calculate_NSSA_cycle_time(n_matrices):
    inputs = np.random.randint(1, 5, (n_matrices, MMU_ROWS, MMU_COLS))
    weights = np.random.randint(1, 5, (n_matrices, MMU_ROWS, MMU_COLS))

    input_shapes = [i.shape for i in inputs]
    output_shapes = [[i.shape[0], w.shape[1]] for i, w in zip(inputs, weights)]

    from NSSA import UnifiedBuffer, MMU, Accumulator, WeightFIFO

    ub = UnifiedBuffer(MMU_ROWS)
    acc = Accumulator(MMU_COLS, ACCUMULATOR_SIZE)
    wf = WeightFIFO(MMU_COLS)
    mmu = MMU(MMU_ROWS, MMU_COLS, ub, wf, acc)

    total_cycles = 0

    for i in range(n_matrices):
        ub.store_input(inputs[i])
        wf.add_weights(weights[i])

    for i in range(n_matrices):
        print(f'NSSA Processing matrix {i+1}/{n_matrices}')
        shape = output_shapes[i]
        cycles = calculate_num_cycles_NSSA(input_shapes[i], shape, MMU_ROWS, None if i == 0 else input_shapes[i-1])

        ub.allocate_output(shape)
        for c in range(cycles):
            mmu.cycle()
        ub.store_acc(acc, shape)

        ground_truth = np.matmul(inputs[i], weights[i])
        result = ub.sram_outputs[-1]
        assert np.array_equal(ground_truth, result)
        total_cycles += cycles

    return total_cycles

if __name__ == '__main__':
    n_matrices = [1, 3, 10, 20]
    cycles_tpu = []
    cycles_NSSA = []
    for n in n_matrices:
        print(f'Testing for {n} matrices')
        cycles_tpu.append(calculate_TPU_cycle_time(n))
        cycles_NSSA.append(calculate_NSSA_cycle_time(n))
    print(cycles_tpu)
    print(cycles_NSSA)
    plt.plot(n_matrices, cycles_tpu, label = "TPU")
    plt.plot(n_matrices, cycles_NSSA, label = "NSSA")
    plt.xlabel('Number of Pipelined Matrices')
    plt.ylabel('Total Number of Cycles')
    plt.legend()
    plt.show()
import numpy as np
from utilities import tile_matrix, calculate_num_cycles_NSSA
from NSSA import UnifiedBuffer, MMU, Accumulator, WeightFIFO

import sys

MMU_ROWS = 10
MMU_COLS = 10
ACCUMULATOR_SIZE = 256

inputMatrix1 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
inputMatrix2 = np.random.randint(1, 10, (MMU_ROWS, MMU_COLS))
weights1 = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))
weights2 = np.random.randint(1, 100, (MMU_ROWS, MMU_COLS))

output_shapes = [[inputMatrix1.shape[0], weights1.shape[1]], [inputMatrix2.shape[0], weights2.shape[1]]]

ub = UnifiedBuffer(MMU_ROWS)
acc = Accumulator(MMU_COLS, ACCUMULATOR_SIZE)
wf = WeightFIFO(MMU_COLS)
mmu = MMU(MMU_ROWS, MMU_COLS, ub, wf, acc)

ub.store_input(inputMatrix1)
ub.store_input(inputMatrix2)

wf.add_weights(weights1)
wf.add_weights(weights2)

cycles1 = calculate_num_cycles_NSSA(inputMatrix1.shape, output_shapes[0], MMU_ROWS)
cycles2 = calculate_num_cycles_NSSA(inputMatrix2.shape, output_shapes[1], MMU_ROWS, inputMatrix1.shape)

ub.allocate_output(output_shapes[0])
ub.allocate_output(output_shapes[1])
mmu.display()
for i in range(cycles1):
    mmu.cycle()
    mmu.display()
    x = input()
    if x == 'q':
        sys.exit(0)
ub.store_acc(acc, output_shapes[0], index=0)
for i in range(cycles2):
    mmu.cycle()
    mmu.display()
    x = input()
    if x == 'q':
        sys.exit(0)
ub.store_acc(acc, output_shapes[1], index=1)

ground_truth1 = np.matmul(inputMatrix1, weights1)
result1 = ub.sram_outputs[0]
assert np.array_equal(ground_truth1, result1)

ground_truth2 = np.matmul(inputMatrix2, weights2)
result2 = ub.sram_outputs[1]
assert np.array_equal(ground_truth2, result2)
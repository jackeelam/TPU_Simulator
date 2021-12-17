import numpy as np
from utilities import tile_matrix, calculate_num_cycles_TPU
from TPU import UnifiedBuffer, MMU, Accumulator, WeightFIFO

import sys

MMU_ROWS = 10
MMU_COLS = 10
ACCUMULATOR_SIZE = 256

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

ub = UnifiedBuffer(MMU_ROWS)
acc = Accumulator(MMU_COLS, ACCUMULATOR_SIZE)
mmu = MMU(MMU_ROWS, MMU_COLS, ub, acc)
wf = WeightFIFO(mmu, input_shapes.copy())

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

ground_truth = np.matmul(inputMatrix, weights)
print('GROUND TRUTH: ')
print(ground_truth)

ub.allocate_output(output_shape)
for i in range(4): # tiled into 4 subsections
    shape = sub_output_shapes[i*2]
    cycles1 = calculate_num_cycles_TPU(input_shapes[i*2], shape, MMU_ROWS, None if i == 0 else input_shapes[i*2-1])
    cycles2 = calculate_num_cycles_TPU(input_shapes[i*2+1], shape, MMU_ROWS, input_shapes[i*2])

    acc.set_acc_cap(shape[0])
    for c in range(cycles1):
        wf.cycle()
        mmu.cycle()
        acc.display()
        x = input()
        if x == 'q':
            sys.exit(0)
    acc.set_acc_cap(acc.acc_size)
    for c in range(cycles2):
        wf.cycle()
        mmu.cycle()
        acc.display()
        x = input()
        if x == 'q':
            sys.exit(0)

    ub.store_acc(acc, shape=shape, start_row=shape[0]*(i//2), start_col=shape[1]*(i%2))

result = ub.sram_outputs[0]
assert np.array_equal(result, ground_truth)
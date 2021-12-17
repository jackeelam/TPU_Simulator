import numpy as np
from TPU import UnifiedBuffer, MMU, Accumulator, WeightFIFO

MMU_ROWS = 10
MMU_COLS = 10
ACCUMULATOR_SIZE = 256

inputMatrix = np.random.randint(1, 5, (MMU_ROWS, MMU_COLS))
weights = np.random.randint(1, 5, (MMU_ROWS, MMU_COLS))

output_shape = [inputMatrix.shape[0], weights.shape[1]]

ub = UnifiedBuffer(MMU_ROWS)
acc = Accumulator(MMU_COLS, ACCUMULATOR_SIZE)
mmu = MMU(MMU_ROWS, MMU_COLS, ub, acc)
wf = WeightFIFO(mmu, [inputMatrix.shape])

ub.store_input(inputMatrix)
wf.add_weights(weights)

ground_truth = np.matmul(inputMatrix, weights)
print('GROUND TRUTH: ')
print(ground_truth)

ub.allocate_output(output_shape)
# for c in range(cycles):
while True:
    wf.cycle()
    mmu.cycle()
    acc.display()
    x = input()
    if x == 'q':
        break
ub.store_acc(acc, output_shape)

result = ub.sram_outputs[-1]
assert np.array_equal(ground_truth, result)
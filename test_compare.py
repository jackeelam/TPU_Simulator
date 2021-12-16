import numpy as np
from test_NSA import createNSA
from test_TPU import createTPU

N_MATRICES = 20
MMU_ROWS = 100
MMU_COLS = 100
ACCUMULATOR_SIZE = 4096

inputs = np.random.randint(0, 5, )

ub, acc, mmu, wf = createTPU([inputMatrix])

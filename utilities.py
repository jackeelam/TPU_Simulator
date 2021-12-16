import numpy as np

def tile_matrix(M, largest_submatrix_size):
    if largest_submatrix_size > len(M):
        largest_submatrix_size = len(M)

    n_A = largest_submatrix_size

    A = M[:n_A, :n_A]
    B = M[:n_A, n_A:]
    C = M[n_A:, :n_A]
    D = M[n_A:, n_A:]

    return A, B, C, D

def calculate_num_cycles_TPU(input_shape, output_shape, mmu_rows, prev_pipeline_input_shape=None):
    if prev_pipeline_input_shape == None:
        offset = mmu_rows - output_shape[1] - 1 # number of cycles it takes for output to get through empty systolic array rows
    else:
        offset =  -np.clip(prev_pipeline_input_shape[1] - input_shape[1], None, 0) - (prev_pipeline_input_shape[0] + prev_pipeline_input_shape[1] - 1)
    return input_shape[0] + input_shape[1] + output_shape[0] - 1 + offset

def calculate_num_cycles_NSSA(input_shape, output_shape, mmu_rows, prev_pipeline_input_shape=None):
    if prev_pipeline_input_shape == None:
        offset = (mmu_rows - output_shape[1]) * 2
    else:
        offset = -(prev_pipeline_input_shape[1] - 1) - (prev_pipeline_input_shape[0] - 2)
    return 2*input_shape[1] + input_shape[0] + output_shape[1] - 2 + offset
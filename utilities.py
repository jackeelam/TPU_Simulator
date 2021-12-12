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
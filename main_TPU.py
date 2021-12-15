from test_TPU import *


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
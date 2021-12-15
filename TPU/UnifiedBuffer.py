import numpy as np
from queue import Queue


class UnifiedBuffer:
    """
    TPU Unified Buffer

    Represents the unified buffer and all inherent SRAM and systolic array setup elements.
    The Static RAM stores inputs and outputs, and performs the necessary conversions to 
    setup the data into a format to feed into the systolic array MMU.
    """

    def __init__(self, systolic_array_rows):
        """
        Initialize the SRAM and systolic array buffer

        Parameters:
            systolic_array_rows: Number of rows in the MMU to define the buffer dimensions
        """
        self.sram = []
        self.sram_outputs = []
        self.systolic_array_size = systolic_array_rows

        self.systolic_array_buffer = []
        for i in range(systolic_array_rows):
            self.systolic_array_buffer.append(Queue()) # init with empty queues

    def update_systolic_array_buffer(self, offset=0):
        """
        Update the systolic array buffer with additional inputs

        Parameters:
            offset: Padding of zeros between the current input and previous input. This is used 
                    to manage situations when the current input is larger than the previous input
        """
        input = self.sram[-1].T
        for row in range(self.systolic_array_size):
            # pad with traingle of zeroes, along with offset padding
            for i in range(np.clip(row - self.systolic_array_buffer[row].qsize(), 0, None) + offset):
                self.systolic_array_buffer[row].put(0)

            # insert input, or fill with zeros
            buffer = np.zeros(input[0].shape[0])
            if row < len(input):
                buffer = input[row]
            for element in buffer:
                self.systolic_array_buffer[row].put(element)

    def display_systolic_array_buffer(self):
        """
        Test function to output current state of systolic array
        """
        for row in self.systolic_array_buffer:
            print(',\t'.join([str(i) for i in row.queue]))

    def allocate_output(self, shape):
        """
        Allocate region of SRAM for the MMU output to be stored. Particularly important when
        the output is tiled and needs to be combined after performing submatrix multiplications
        """
        self.sram_outputs.append(np.zeros(shape, dtype=np.int16))

    def store_acc(self, accumulator, shape, index=0, start_row=0, start_col=0):
        """
        Extract a region of the accumulators and store it in the correct region of the specified 
        pre-allocated SRAM region
        """
        mat = accumulator.get_block(shape)
        self.sram_outputs[index][start_row:start_row+mat.shape[0], start_col:start_col+mat.shape[1]] = mat

    def store_input(self, input):
        """
        Calculate the padding and store the new input into the SRAM and update the systolic array
        buffer. Determines the offset which is the padding of zeros between the current input and 
        previous input. This is used to manage situations when the current input is larger than the 
        previous input
        """
        offset = 0 if len(self.sram) == 0 else np.clip(input.shape[0] - self.sram[-1].shape[0], 0, None)
        input = input.T
        for row in range(self.systolic_array_size):
            # pad with traingle of zeroes, along with offset padding
            for i in range(np.clip(row - self.systolic_array_buffer[row].qsize(), 0, None) + offset):
                self.systolic_array_buffer[row].put(0)

            # insert input, or fill with zeros
            buffer = np.zeros(input[0].shape[0])
            if row < len(input):
                buffer = input[row]
            for element in buffer:
                self.systolic_array_buffer[row].put(element)

        self.sram.append(input)

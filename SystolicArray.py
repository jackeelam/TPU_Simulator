from queue import Queue
import numpy as np
from FIFOWeightBuffer import FIFOWeightBuffer
from UnifiedInputBuffer import UnifiedInputBuffer

#Sample input and weights
input=[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]

weights = [
    [10,20,30],
    [40,50,60],
    [70,80,90] 
]

matrix_multiply_unit = np.zeros(shape=(256,256))

class MultiplyUnit:
  def __init__(self, weight_buffer, input_buffer):
    self.weight_buffer = weight_buffer
    self.input_buffer = input_buffer
    self.mxu = np.zeros(shape=(256,256))

  #Specify the dimension of weights you want to take out of the FIFO weight buffer and fill the mxu 256x256 array with
  def fill_with_weights(self,num_rows, num_cols):
    weights_to_mxu = self.weight_buffer.get_weights(num_rows, num_cols)
    for row in range(num_rows):
      for col in range(num_cols):
        self.mxu[row][col] = weights_to_mxu[row][col]

  #Get the next col from unified buffer to process
  #col = self.input_buffer.get_next_col_input()
    
#Initialize input buffer with the sample input and test out appending inputs
ib = UnifiedInputBuffer(input)
ib.append_input(input)

print("Unified Input Buffer")
print(ib.input_buffer)
#See if value of col is correct
# print(ib.get_next_col_input())

#Extract by column the input
for q in ib.input_buffer:  
  print(q.queue)

#Initialize weight buffer
wb = FIFOWeightBuffer(weights)
print(wb.weight_buffer)

#Fill mxu with weights
mxu = MultiplyUnit(wb, ib)
mxu.fill_with_weights(3,3)

#Print out what mxu looks like currently
print(mxu.mxu)
import numpy as np
from queue import Queue

#Weight FIFO buffer
class FIFOWeightBuffer:
  def __init__(self, weights):
    self.weight_rows = len(weights)
    self.weight_buffer = self.initialize_weight_buffer(weights, self.weight_rows)

  #Using an initial weight matrix
  def initialize_weight_buffer(self, weights, weight_rows):
    #Initialize input_buffer of size of original input
    weight_buffer = []
    for row in range(weight_rows):
      weight_buffer.append(Queue())

      for element in weights[row]:
          weight_buffer[row].put(element)
          
    return weight_buffer

  #Append more weights to weight buffer
  def append_weights(self, additional_weights):
    for row in range(self.weight_rows):

      #insert input
      for element in additional_input[row]:
        self.weight_buffer[row].put(element)

  #Remove weights from weight buffer given dimensions
  def get_weights(self, num_rows, num_cols):
    weights = np.zeros(shape=(num_rows, num_cols))
    for row in range(num_rows):
      for col in range(num_cols):
        weights[row][col]=self.weight_buffer[row].get()

    return weights
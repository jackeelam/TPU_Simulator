import numpy as np
from queue import Queue

#Weight FIFO buffer
class WeightFIFO:
  def __init__(self, weights):
    self.weight_rows = len(weights)
    self.weight_cols = len(weights[0])
    self.weight_buffer = self.initialize_weight_buffer(weights, self.weight_cols, self.weight_rows)

  #Using an initial weight matrix
  def initialize_weight_buffer(self, weights, weight_cols, weight_rows):
    #Initialize input_buffer of size of original input
    weight_buffer = []

    for col in range(weight_cols):
        weight_buffer.append(Queue())
    
        #pad with triangles
        for i in range(col):
            weight_buffer[col].put(0)

        for row in range(weight_rows):
            weight_buffer[col].put(weights[row][col])

          
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
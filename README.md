# Cycle-level Simulation of Tensor Processing Unit (TPU)
ECE 209AS: AI on Chip final project

### Abstract
With the stagnation of Moore’s Law, it has become increasingly more pertinent to use domain specific architectures such as the Tensor Processing Unit(TPU) that Google created for accelerating the computations involved with Deep Neural Networks(DNNs). The most important computation in a DNN involves matrix multiplication due to the ability to convert to it from convolution. Matrix multiplication is implemented via systolic arrays which are 2D structures composed of multiply accumulators(MACs) which process inputs and weights and pass that information to other MACs. The two most popular systolic array structures that perform matrix multiplication include the Non-Stationary Weight Systolic Array(NSSA) and the TPU-style Stationary Weight Systolic Array(TSSA). The architecture of both are simulated and their performance is compared, to which we see the TSSA significantly outperforms the NSSA.

### How to run
Please refer to the demo files in the demo/ folder for examples on how to run both the TPU-style and NSSA architectures. 


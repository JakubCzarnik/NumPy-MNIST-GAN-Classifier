from abc import abstractmethod
import numpy as np
from learning.tensor import Tensor


class Layer:
   @abstractmethod
   def __init__(self, input_size, output_size, activation=None):
      pass

   @abstractmethod
   def forward(self, inputs):
      pass

   @abstractmethod
   def update_parameters(self, learning_rate):
      pass
   
 
class Dense(Layer):
   def __init__(self, 
                input_size=1, 
                output_size=1, 
                activation=None,
                kernel_initializer=None):
      self.input_size = input_size
      self.output_size = output_size
      self.activation = activation

      if kernel_initializer is None:
         w = np.random.uniform(-1e-3, 1e-3, (self.input_size, self.output_size))
         self.weights = Tensor(w, weight=True, is_watched=True)

      elif isinstance(kernel_initializer, Tensor):
         self.weights = kernel_initializer

      else:
         self.weights = Tensor(kernel_initializer,  
                               weight=True, 
                               is_watched=True)
      self.bias = Tensor(np.zeros(self.output_size))


   def forward(self, inputs):
      output = inputs.dot(self.weights) + self.bias

      if self.activation == "sigmoid":
         output = output.sigmoid()
      elif self.activation == "tanh":
         output =  output.tanh()
      elif self.activation == "leakyrelu":
         output = output.leakyrelu()
      elif self.activation == "relu":
        output = output.relu()
      elif self.activation == "softmax":
        output = output.softmax()

      return output 
   

   def update_parameters(self, learning_rate):
      self.weights -= learning_rate * self.weights.grad
      self.bias -= learning_rate * self.weights.bias_grad
      

   def __repr__(self):
      return f"AffineLayer | Inputs {self.input_size} | "+\
           f"outputs {self.output_size} | Params {self.output_size * (self.input_size + 1)}"


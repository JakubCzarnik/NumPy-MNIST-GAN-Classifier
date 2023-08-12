import numpy as np

class GradientTape:
   def __enter__(self):
      Tensor._inscope = True
      return self

   def __exit__(self, *args):
      Tensor._inscope = False


class Tensor(np.ndarray):
   _inscope = False
   _layers = [] # for reseting gradients

   def __new__(cls,
               input_array, 
               dtype=np.float32, 
               is_watched=False,
               local_gradients=(),
               weight=False):
      obj = np.asarray(input_array, dtype=dtype).view(cls)
      obj._watched = is_watched
      obj.local_gradients = local_gradients
      if weight:
         cls._layers.append(obj)
      return obj


   def __array_finalize__(self, obj):
      if obj is None:
         return
      self._watched = getattr(obj, 'is_watched', None)
      self.local_gradients = getattr(obj, 'local_gradients', None)
      self.grad = 0 # weights grad
      self.bias_grad = 0 # bias grad


   def backward(self):
      Tensor.zero_grad() # only layers
      self._backward()


   def _backward(self, path_value=1):
      for child_var, gradient, is_dot_prod in self.local_gradients:
         if not is_dot_prod:
         # child_var, gradient, dot_pr = (child_var, gradient, false)
         # self = output of a function
            weights_grad = np.multiply(gradient, path_value)
            child_var.grad += weights_grad
            child_var._backward(weights_grad)
         else:
         # child_var, gradient, dot_pr = (weights, input to weights, true)
         # self = output of a layer | (input to weights) o (weights)
            weights_grad = np.dot(gradient.T, path_value) 
            
            child_var.grad += weights_grad
            child_var.bias_grad += np.sum(path_value, axis=0)

            grad_out = np.dot(path_value, child_var.T)
            gradient._backward(grad_out)


   def to_numpy(self):
      return np.copy(self)


   def copy(self, *args):
      return Tensor(np.copy(self, *args))


   @classmethod
   def zero_grad(cls):
      for layer in cls._layers:
         layer.grad = 0
         layer.bias_grad = 0


   def multiply(self, other):
      return self.__mul__(other)
   

   def dot(self, other):
      result = np.dot(self, other)

      if not (self._inscope and other._watched):
         return Tensor(result)
      # only d/dy is calculated
      # weights, inputs, dot=True
      grads = [(other, self, True)]

      return Tensor(result, is_watched=True, local_gradients=grads)


   def __neg__(self):
      result = np.negative(self)

      if not (self._inscope and self._watched):
         return Tensor(result)
      
      dx = np.full_like(self, -1)
      grads = (self, dx, False)

      return Tensor(result, is_watched=True, local_gradients=grads)
     

   def __add__(self, other):
      result = np.add(self, other)

      if not isinstance(other, Tensor):
         other = Tensor(other)

      if not self._inscope:
         return Tensor(result)
      if not (self._watched or other._watched):
         return Tensor(result)

      grads = []
      if self._watched:
         dx = np.ones_like(self)
         grads.append((self, dx, False))
      if other._watched:
         dy = np.ones_like(other)
         grads.append((other, dy, False))

      return Tensor(result, is_watched=True, local_gradients=grads)
   

   def __radd__(self, other):
      return self.__add__(other)

   
   def __sub__(self, other):
      result = np.subtract(self, other)

      if not isinstance(other, Tensor):
         other = Tensor(other)

      if not self._inscope:
         return Tensor(result)
      if not (self._watched or other._watched):
         return Tensor(result)

      grads = []
      if self._watched:
         dx = np.ones_like(self)
         grads.append((self, dx, False))
      if other._watched:
         dy = np.full_like(self, -1)
         grads.append((other, dy, False))
 
      return Tensor(result, is_watched=True, local_gradients=grads)


   def __rsub__(self, other):
     other = Tensor(other)
     return other.__sub__(self)


   def __mul__(self, other):
      result = np.multiply(self, other)

      if not isinstance(other, Tensor):
         other = Tensor(other)

      if not self._inscope:
         return Tensor(result)
      if not (self._watched or other._watched):
         return Tensor(result)
      
      if self._watched and other._watched and (self.shape != other.shape):
         raise ValueError("Can't compute the derivatives of both matrices without using reverse broadcasting")
      
      grads = []
      if self._watched:
         dx = np.broadcast_to(other, self.shape)
         grads.append((self, dx, False))
      if other._watched:
         dy = np.broadcast_to(self, other.shape)
         grads.append((other, dy, False))

      return Tensor(result, is_watched=True, local_gradients=grads)
   

   def __rmul__(self, other):
      return self.__mul__(other)


   def __truediv__(self, other):
      eps = np.where(other >= 0, 1e-6, -1e-6)
      inverse = np.divide(1, other+eps)

      if isinstance(other, Tensor):
         inverse = Tensor(inverse, 
                          is_watched=other._watched,
                          local_gradients=other.local_gradients)
      # a/b = a * (1/b)
      return self.__mul__(inverse)


   def __rtruediv__(self, other):
      other = Tensor(other)
      return other.__truediv__(self)


   def __pow__(self, other):
      x = self.to_numpy()
      y = other   
      eps = np.where(x >= 0, 1e-6, -1e-6)

      if isinstance(other, Tensor):
         y = other.to_numpy()
      else:
         other = Tensor(other)
         y = other

      result = x**y

      if not self._inscope:
         return Tensor(result)
      if not (self._watched or other._watched):
         return Tensor(result)

      grads = []
      if self._watched:
         dx = result * (y / (x+eps))
         grads.append((self, dx, False))
      if other._watched:
         dy = result * np.log(x)
         grads.append((other, dy, False))
      
      return Tensor(result, is_watched=True, local_gradients=grads)


   def __sum__(self, *args, **kwargs):
      result = super().sum(*args, **kwargs)
      
      if not (self._inscope and self._watched):
         return Tensor(result)

      dx = np.ones_like(self)
      grads = [(self, dx, False)]

      return Tensor(result, is_watched=True, local_gradients=grads)


   def sum(self, *args, **kwargs):
      return self.__sum__(*args, **kwargs)

   
   def mean(self, *args, **kwargs):
      x = self.to_numpy()
      result =  np.mean(x, *args, **kwargs)

      if not (self._inscope and self._watched):
         return Tensor(result)
      
      dx = np.ones_like(self) / self.size
      grads = [(self, dx, False)]

      return Tensor(result, is_watched=True, local_gradients=grads)
 

   def sigmoid(self):
      x = self.to_numpy()
      sig = 1 / (1 + np.exp(-x))
      result = sig

      if not (self._inscope and self._watched):
         return Tensor(result)

      dx = sig*(1-sig)
      grads = [(self, dx, False)]

      return Tensor(result, is_watched=True, local_gradients=grads)


   def tanh(self):
      x = self.to_numpy()
      tanh = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
      result = tanh # (e^(2x)-1) / (e^(2x)+1)

      if not (self._inscope and self._watched):
         return Tensor(result)

      dx = 1 - tanh**2
      grads = [(self, dx , False)]

      return Tensor(result, is_watched=True, local_gradients=grads)


   def relu(self):
      x = self.to_numpy()
      result = np.maximum(0, x)

      if not (self._inscope and self._watched):
         return Tensor(result)

      dx = np.where(x > 0, 1, 0)
      grads = [(self, dx, False)]

      return Tensor(result, is_watched=True, local_gradients=grads)
   

   def leakyrelu(self, alpha=0.2):
      x = self.to_numpy()
      result = np.maximum(alpha * x, x)

      if not (self._inscope and self._watched):
         return Tensor(result)

      dx = np.where(x > 0, 1, alpha)
      grads = [(self, dx, False)]

      return Tensor(result, is_watched=True, local_gradients=grads)
   

   def binary_crossentropy(self, y_true):
      y_pred = self.to_numpy()
      other = y_true
      if isinstance(y_true, Tensor):
         y_true = y_true.to_numpy()
      else:
         other = Tensor(y_true)

      eps = 1e-7
      y_pred = np.clip(y_pred, eps, 1 - eps)

      loss = -(y_true * np.log(y_pred) + 
               (1 - y_true + eps) * np.log(1 - y_pred + eps))
 
      if np.any(loss<0):
         print("WARNING! negative loss in binary_crossentropy")
      result = loss.mean()


      if not self._inscope:
         return Tensor(result)
      if not (self._watched) or other._watched:
         return Tensor(result)

      grads = []
      if self._watched:
         dx = (y_pred - y_true) / (y_pred - y_pred**2)
         grads.append((self, dx, False))
      if other._watched:
         dy = -np.log(y_pred) + np.log(1 - y_pred)
         grads.append((other, dy, False))
      return Tensor(result, is_watched=True, local_gradients=grads)
   

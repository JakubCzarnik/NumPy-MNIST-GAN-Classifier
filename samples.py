import numpy as np
from learning.tensor import Tensor, GradientTape

## grads supports:
# +, -, *, /, **
# .sum .mean .dot .sigmoid .tanh .relu .leakyrelu .binarycrossentropy


### 1. basic derivatives
print("Basic derivatives:\n")
x = Tensor(3, is_watched=True) # if is_watche=False, then grads will not be calculated for x
y = Tensor(4, is_watched=True)

with GradientTape():
   c = x*y
   d = c**2


d.backward()
d.backward()
print(f"{x.grad = :.1f}") # d/dx [(x*y)**2] = 2(x*y)*y = 96.
print(f"{y.grad = :.1f}") # d/dy [(x*y)**2] = 2(x*y)*x = 72.
print(f"{c.grad = :.1f}") # d/dc [c**2]     = 2(c)     = 24.
print(f"{d.grad = :.1f}") # d/dd [d]        = (144)'   = 0.

del x, y, c, d



### 2. Layers derivatives
print("\n\nLayer Training:\n")

# Create layers
w1 = np.random.uniform(-1, 1, (10, 5))
w1 = Tensor(w1, weight=True, is_watched=True)
b1 = Tensor(np.zeros(5))

w2 = np.random.uniform(-1, 1, (5, 1))
w2 = Tensor(w2, weight=True, is_watched=True)
b2 = Tensor(np.zeros(1))

# Create target and random input (6 batches)

input = Tensor(np.random.uniform(-1, 1, (6, 10)))
target = Tensor(np.array([1, .8, .6, .4, .2, 0]).reshape(-1, 1))

for i in range(5):
   with GradientTape():
      output1 = input.dot(w1) + b1
      output2 = output1.dot(w2) + b2

      (output2*2)**2
      loss = ((output2-target)**2).mean()

   print(f"Iter: {i}  {loss = :.3f}")
   loss.backward()

   # Update weights
   lr = 0.1
   w1 -= lr*w1.grad
   b1 -= lr*w1.bias_grad
   w2 -= lr*w2.grad
   b2 -= lr*w2.bias_grad




### 3. Layers with activation
print("\n\nLayer Training with activations:\n")


w1 = np.random.uniform(-1, 1, (10, 5))
w1 = Tensor(w1, weight=True, is_watched=True)
b1 = Tensor(np.zeros(5))

w2 = np.random.uniform(-1, 1, (5, 1))
w2 = Tensor(w2, weight=True, is_watched=True)
b2 = Tensor(np.zeros(1))

input = Tensor(np.random.uniform(-1, 1, (6, 10)))
target = Tensor(np.array([1, 1, 0, 0, 1, 0]).reshape(-1, 1))

for i in range(5):
   with GradientTape():
      output1 = input.dot(w1) + b1
      output1 = output1.tanh()
      output2 = output1.dot(w2) + b2
      output2 = output2.sigmoid()

      loss = output2.binary_crossentropy(target)

   print(f"Iter: {i}  {loss = :.3f}")
   loss.backward()

   # Update weights
   lr = 0.5
   w1 -= lr*w1.grad
   b1 -= lr*w1.bias_grad
   w2 -= lr*w2.grad
   b2 -= lr*w2.bias_grad


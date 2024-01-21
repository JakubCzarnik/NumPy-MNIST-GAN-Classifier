import numpy as np
from learning.tensor import Tensor, GradientTape
from learning.layer import Dense
from utils import *

class Classifier:
   def __init__(self, input_size, output_size, learning_rate=0.005):
      self.input_size = input_size
      self.output_size = output_size
      self.learning_rate = learning_rate

      self.layer1 = Dense(input_size, 392, activation="sigmoid", kernel_initializer=np.random.uniform(-1e-2, 1e-2, (input_size, 392)))
      self.layer2 = Dense(392, output_size, activation="sigmoid", kernel_initializer=np.random.uniform(-1e-2, 1e-2, (392, output_size)))


   def __call__(self, inputs):
      x = self.layer1.forward(inputs)
      x = self.layer2.forward(x)
      return x


   def update_parameters(self):
      self.layer1.update_parameters(self.learning_rate)
      self.layer2.update_parameters(self.learning_rate)



# Load Data
X_train, y_train, X_val, y_val = load_data_classifier('MNIST')

# Hyperparameters
epochs = 1000
lr = 0.008
batch_size = 50


model = Classifier(input_size=784, output_size=10, learning_rate=lr)


# Training
n_batches = X_train.shape[0] // batch_size
for epoch in range(epochs):
   print(f"Epoch {epoch}/{epochs}")
   t_loss = v_loss = t_acc = v_acc = 0

   for batch in range(0, n_batches):
      batch_x = X_train[batch*batch_size: (batch+1)*batch_size]
      batch_y = y_train[batch*batch_size: (batch+1)*batch_size]

      batch_x = Tensor(batch_x)
      with GradientTape():
         # Calculations are stored within the Tensor objects, not within the GradientTape itself.
         # Inside this context, the Tensor records the defined calculations within itself.

         output = model(batch_x)
         loss = ((batch_y - output)**2).sum()

      loss.backward()
      model.update_parameters()
      # Calculate Training Metrics
      output = np.argmax(output, axis=1)
      batch_y = np.argmax(batch_y, axis=1)
      train_accuracy = np.mean(np.where(output==batch_y, 1, 0))

      alpha = 1/(batch+1) if batch != 0 else 1

      t_loss = (1-alpha)*t_loss + alpha*loss
      t_acc = (1-alpha)*t_acc + alpha*train_accuracy

      print(f"batch: {batch}/{n_batches}  {t_loss = :.3f}  {t_acc = :.2%}", end="\r")

   if epoch==100:
      model.learning_rate *= 0.7
   ### Visualize ###
   # show_from_file(model, file="test.png")
   # show_from_dataset(model, test_x)
   create_confusion_matrix(model, X_val, y_val, batch_size, epoch)
     
   v_loss, v_acc = evaluate(model, X_val, y_val, batch_size)
   print(f"epoch: {epoch}  {t_loss = :.3f}  {t_acc = :.2%}  {v_loss = :.3f}  {v_acc = :.2%}   ")

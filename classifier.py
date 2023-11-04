import numpy as np
from learning.tensor import Tensor, GradientTape
from learning.layer import Dense
from mnist import MNIST
import matplotlib.pyplot as plt
import cv2

class Classifier:
   def __init__(self, input_size, output_size, learning_rate=0.005):
      self.input_size = input_size
      self.output_size = output_size
      self.learning_rate = learning_rate

      self.layer1 = Dense(input_size, 392, activation="sigmoid")
      self.layer2 = Dense(392, output_size, activation="sigmoid")


   def __call__(self, inputs):
      x = self.layer1.forward(inputs)
      x = self.layer2.forward(x)
      return x


   def update_parameters(self):
      self.layer1.update_parameters(self.learning_rate)
      self.layer2.update_parameters(self.learning_rate)


def show_from_file(model, file="test.png"):
   """Shows prediction on custom image."""
   image = cv2.imread(file) # I used paint to generate new samples 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
   inputs = image.reshape((1, 784))
   output = model(inputs)


   idx = np.argmax(output, axis=1)[0]
   print(f"Number: {idx}, confident: {output[0][idx]*100:.2f}%")
   plt.imshow(image, cmap="hot")
   plt.show()


def show_from_dataset(model, x_test):
   import matplotlib.pyplot as plt
   output = model(x_test[0:5])
   images = x_test[0:5].reshape((5, 28,28))
   for i, img in enumerate(images):
      plt.imshow(img, cmap="hot")
      idx = np.argmax(output[i], axis=0)
      print(f"Number: {idx}, confident: {output[i][idx]*100:.2f}%")
      print(np.round(output[i]*100,3))
      plt.show()


def evaluate(model, X_val, y_val, batch_size):
   """Evaluates model on all validation dataset"""
   val_loss = 0
   val_accuracy = 0

   n_batches = X_val.shape[0]//batch_size
   for i in range(n_batches):
      val_x = X_val[i*batch_size: (i+1)*batch_size]
      val_y = y_val[i*batch_size: (i+1)*batch_size]
      val_y_hot = one_hot_encoder(val_y, 10)

      val_output_hot = model(val_x)
      val_output = np.argmax(val_output_hot, axis=1)

      loss = ((val_y_hot - val_output_hot)**2).sum()
      accuracy = np.mean(np.where(val_output==val_y, 1, 0))

      # calculate mean
      alpha = 1/(i+1) if i != 0 else 1

      val_loss = (1-alpha)*val_loss + alpha*loss
      val_accuracy = (1-alpha)*val_accuracy + alpha*accuracy
   return val_loss, val_accuracy


def one_hot_encoder(data, n_classes):
   one_hot_encoded = np.zeros((len(data), n_classes))
   one_hot_encoded[np.arange(len(data)), data] = 1
   return one_hot_encoded


# Load Data
mndata = MNIST('MNIST')
train_x, train_y = mndata.load_training()
val_x, val_y = mndata.load_testing()


train_x = 2*np.array(train_x)/255 - 1
train_y = np.array(train_y)

val_x = 2*np.array(val_x)/255 - 1
val_y = np.array(val_y)

train_y = one_hot_encoder(train_y, 10)

# Hyperparameters
epochs = 5000
lr = 0.01
batch_size = 50


model = Classifier(input_size=784, output_size=10, learning_rate=lr)


# Training
n_batches = train_x.shape[0] // batch_size
for epoch in range(epochs):
   print(f"Epoch {epoch}/{epochs}")
   t_loss = v_loss = t_acc = v_acc = 0

   for batch in range(0, n_batches):
      batch_x = train_x[batch*batch_size: (batch+1)*batch_size]
      batch_y = train_y[batch*batch_size: (batch+1)*batch_size]

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

   if epoch==200:
      model.learning_rate /= 2
   ### Visualize ###
   # show_from_file(model, file="test.png")
   # show_from_dataset(model, test_x)
     
   v_loss, v_acc = evaluate(model, val_x, val_y, batch_size)
   print(f"epoch: {epoch}  {t_loss = :.3f}  {t_acc = :.2%}  {v_loss = :.3f}  {v_acc = :.2%}   ")

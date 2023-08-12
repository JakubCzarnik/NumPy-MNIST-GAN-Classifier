import numpy as np
from learning.tensor import Tensor, GradientTape
from learning.layer import Dense
from mnist import MNIST


class Classifier:
   def __init__(self, input_size, output_size, learning_rate=0.005):
      self.input_size = input_size
      self.output_size = output_size
      self.learning_rate = learning_rate

      self.layer1 = Dense(input_size, 100, activation="sigmoid")
      self.layer2 = Dense(100, output_size, activation="sigmoid")


   def __call__(self, inputs):
      x = self.layer1.forward(inputs)
      x = self.layer2.forward(x)
      return x


   def update_parameters(self):
      self.layer1.update_parameters(self.learning_rate)
      self.layer2.update_parameters(self.learning_rate)



def one_hot_encoder(data, n_classes):
   one_hot_encoded = np.zeros((len(data), n_classes))
   one_hot_encoded[np.arange(len(data)), data] = 1
   return one_hot_encoded


# Load Data
mndata = MNIST('MNIST')
train_x, train_y = mndata.load_training()
test_x, test_y = mndata.load_testing()


train_x = 2*np.array(train_x)/255 - 1
train_y = np.array(train_y)

test_x = 2*np.array(test_x)/255 - 1
test_y = np.array(test_y)

train_y = one_hot_encoder(train_y, 10)

# Hyperparameters
epochs = 40
lr = 0.005
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

      # Calculate Validation Metrics
      idx = np.random.choice(test_x.shape[0]//n_batches-1, 1)[0]
      val_x = test_x[idx*batch_size: (idx+1)*batch_size]
      val_y = test_y[idx*batch_size: (idx+1)*batch_size]
      val_y_hot = one_hot_encoder(val_y, 10)

      val_output_hot = model(val_x)
      val_output = np.argmax(val_output_hot, axis=1)

      val_loss = ((val_y_hot - val_output_hot)**2).sum()
      val_accuracy = np.mean(np.where(val_output==val_y, 1, 0))

      # Update Displayed Metrics
      alpha = 1/(batch+1)

      t_loss = (1-alpha)*t_loss + alpha*loss
      v_loss = (1-alpha)*v_loss + alpha*val_loss

      t_acc = (1-alpha)*t_acc + alpha*train_accuracy
      v_acc = (1-alpha)*v_acc + alpha*val_accuracy


      print(f"batch: {batch}/{n_batches}  {t_loss = :.3f}  {t_acc = :.2%}  {v_loss = :.3f}  {v_acc = :.2%}", end="\r")

   print(f"epoch: {epoch}  {t_loss = :.3f}  {t_acc = :.2%}  {v_loss = :.3f}  {v_acc = :.2%}   ")


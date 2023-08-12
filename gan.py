import numpy as np
import matplotlib.pyplot as plt
from learning.tensor import Tensor, GradientTape
from learning.layer import Dense
from mnist import MNIST


class Generator:
   def __init__(self, input_size, output_size, learning_rate=0.01):
      self.learning_rate = learning_rate

      self.layer1 = Dense(input_size, 200, activation="tanh")
      self.layer2 = Dense(200, 400, activation="tanh")
      self.layer3 = Dense(400, output_size, activation="tanh")


   def __call__(self, inputs):
      x = self.layer1.forward(inputs)
      x = self.layer2.forward(x)
      x = self.layer3.forward(x)
      return x


   def update_parameters(self):
      self.layer1.update_parameters(self.learning_rate)
      self.layer2.update_parameters(self.learning_rate)
      self.layer3.update_parameters(self.learning_rate)
    

class Discriminator:
   def __init__(self, input_size, output_size, learning_rate=0.01):
      self.learning_rate = learning_rate

      self.layer1 = Dense(input_size, 400, activation="leakyrelu")
      self.layer2 = Dense(400, 50, activation="leakyrelu")
      self.layer3 = Dense(50, output_size, activation="sigmoid")


   def __call__(self, inputs):
      x = self.layer1.forward(inputs)
      x = self.layer2.forward(x)
      x = self.layer3.forward(x)
      return x
   

   def update_parameters(self):
      self.layer1.update_parameters(self.learning_rate)
      self.layer2.update_parameters(self.learning_rate)
      self.layer3.update_parameters(self.learning_rate)


# Load Data
mndata = MNIST('MNIST')
train_x, _ = mndata.load_training()
train_x = 2*np.array(train_x)/255 - 1


samples_folder = "image_samples"

# Hyperparameters
lat_dim = 100
epochs = 75
batch_size = 100
g_lr = 0.0003
d_lr = 0.0003


generator = Generator(input_size=lat_dim,
                      output_size=train_x.shape[1], 
                      learning_rate=g_lr)

discriminator = Discriminator(input_size=train_x.shape[1], 
                              output_size=1, 
                              learning_rate=d_lr)

# Generate Static Data
ones = np.ones(shape=(batch_size, 1))
zeros =  np.zeros(shape=(batch_size, 1))
sample_noise = np.random.uniform(low=-1, high=1, size=(batch_size, lat_dim))


# Training
n_batches = (train_x.shape[0] // batch_size)
for epoch in range(epochs):
   print(f"Epoch {epoch}/{epochs}")

   g_logs = d_logs = 0
   for batch in range(n_batches):
      # load inputs for discriminator and generator
      noise = np.random.uniform(low=-1, 
                                high=1, 
                                size=(batch_size, lat_dim))
      true_img = train_x[batch*batch_size: (batch+1)*batch_size]

      noise, true_img = Tensor(noise), Tensor(true_img)
      with GradientTape():
         # Calculations are stored within the Tensor objects, not within the GradientTape itself.
         # Inside this context, the Tensor records the defined calculations within itself.

         fake_img = generator(noise)

         fake_output = discriminator(fake_img)
         real_output = discriminator(true_img)

         # Discriminator Loss
         d_fake_loss = fake_output.binary_crossentropy(zeros)
         d_true_loss = real_output.binary_crossentropy(ones)
         d_loss = (d_fake_loss + d_true_loss)/2

         # Generator Loss
         g_loss = fake_output.binary_crossentropy(ones)

      # Update Generator Weights
      g_loss.backward()
      generator.update_parameters()

      # Update Discriminator Weights
      if not batch%3 == 0:
         d_loss.backward()
         discriminator.update_parameters()

      # Calculate Mean Metrics And Display Them
      alpha = 1/(batch+1)
      d_logs = (1-alpha)*d_logs + (alpha)*np.mean(d_loss)
      g_logs = (1-alpha)*g_logs + (alpha)*np.mean(g_loss)
      print(f"Batch {batch}/{n_batches} dloss {d_logs:.3f}, gloss {g_logs:.3f}", end="\r")

   print(f"Batch {n_batches}/{n_batches} dloss {d_logs:.3f}, gloss {g_logs:.3f}   ")

   # Decrease Learning Rate After 3 Epochs
   if epoch == 2:
      generator.learning_rate *=  0.8
      discriminator.learning_rate *= 0.5

   # Save Samples
   if epoch%5 == 0:
      imgs = generator(sample_noise)

      selected_indices = range(len(imgs))
      fig, axes = plt.subplots(10, 10, figsize=(28, 28))
      fig.subplots_adjust(hspace=0, wspace=0)

      for j, ax in enumerate(axes.flat):
         idx = selected_indices[j]
         img = imgs[idx].reshape(28, 28)
         ax.imshow(img, cmap='gray')
         ax.axis('off')

      output_path = f"{samples_folder}/img_{epoch}.png"
      plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
      plt.clf()
      plt.close()
   
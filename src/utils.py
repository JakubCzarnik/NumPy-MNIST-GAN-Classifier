import cv2
import matplotlib.pyplot as plt, seaborn as sn
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
from mnist import MNIST

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


def create_confusion_matrix(model, X_val, y_val, batch_size, epoch=0):
   all_predictions = []

   n_batches = X_val.shape[0]//batch_size
   for i in range(n_batches):
      val_x = X_val[i*batch_size: (i+1)*batch_size]

      y_pred = model(val_x)
      y_pred = np.argmax(y_pred, axis=1)
      all_predictions.append(y_pred)
   all_predictions = np.array(all_predictions)
   all_predictions = all_predictions.reshape((-1))
   cm = confusion_matrix(y_val, all_predictions)

   df_cm = pd.DataFrame(cm, index=[i for i in "0123456789"],
                     columns=[i for i in "0123456789"])

   plt.figure(figsize=(5, 4))
   sn.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")
   plt.xlabel("Predicted")
   plt.ylabel("Real")
   plt.title("Confusion Matrix")
   plt.savefig(f"image_samples/{epoch}")
   plt.close()


def load_data_classifier(file='MNIST'):
   mndata = MNIST(file)
   X_train, y_train = mndata.load_training()
   X_val, y_val = mndata.load_testing()

   X_train = 2*np.array(X_train)/255 - 1
   y_train = np.array(y_train)

   X_val = 2*np.array(X_val)/255 - 1
   y_val = np.array(y_val)

   y_train = one_hot_encoder(y_train, 10)
   return X_train, y_train, X_val, y_val


def load_data_gan(file='MNIST'):
   mndata = MNIST(file)
   X_train, _ = mndata.load_training()
   # normalize
   X_train = np.array(X_train)/127.5 - 1
   return X_train
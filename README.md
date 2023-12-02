# MNIST Generative Adversarial Network (GAN) and Classifier using NumPy

This repository contains the implementation of a Generative Adversarial Network (GAN) and a classifier for the MNIST dataset using only the NumPy library.

## Project Overview

The primary objective of this project was to build both a GAN and a classifier from scratch, demonstrating their functionality without relying on external deep learning frameworks. The project, created for educational purposes, focuses on the generation of realistic handwritten digit images and accurate classification using NumPy.



## Contents

- `gan.py`: The GAN implementation, generating synthetic handwritten digit images from random noise.
- `classifier.py`: The classifier implementation that categorizes MNIST digit images.
-  `samples.py`: This file contains an example of using automatic differentiation in practice.
- `learning/tensor.py`: An extended NumPy array class with automatic calculation of specific derivatives, enhancing numerical operations and enabling gradient calculations.
- `learning/layer.py`: Implementation of a simple Dense layer class for neural networks.


## Usage
Before running the scripts, please ensure you have downloaded the MNIST dataset.

1. Run `gan.py` to train and generate images using the GAN.
2. Run `classifier.py` to train the classifier and evaluate its performance.


Adjust hyperparameters and settings in the respective scripts to experiment with different configurations.


## Dependencies

The project solely relies on the NumPy library for numerical computations. No external deep learning frameworks are used.

## Training Results
##### Generative Adversarial Network .
Despite the use of dense layers instead of CNNs and the lack of regularization, you can see clear numbers at the end. I also noticed during the training that the model tends to mode colappse, generating more and more ones, which basically seems like a logical decision since it is the simplest number to generate.

![GAN Training](readme_files/gan_training.gif)
##### Classifier.
The model was evaluated on the entire validation dataset (10,000 images) at each epoch. In the 470 epoch, the model achieved an incredible 98.3% accuracy on validation data. Which is a bit surprising because I used sigmoid activation instead of popular softmax in multiclass classification problem.

![Classifier Training](readme_files/classifier.png)
## License

This project is licensed under the [MIT License](LICENSE).

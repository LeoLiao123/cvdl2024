# NCKU CVDL2024 Hw2

## Overview

This is the assignment 2 of National Cheng Kung University Computer Vision and Deep Learning course, implementing deep learning tasks including training a VGG19 model on CIFAR-10, hyperparameter optimization with Optuna, model evaluation, and implementing a DCGAN to generate MNIST-like images.

## Features

1. **Train VGG19 Model**

   - Train a VGG19 model on the CIFAR-10 dataset using SGD and Adam optimizers.
   - Log training progress and save the best model.

2. **Hyperparameter Optimization**

   - Use Optuna to optimize hyperparameters such as learning rate, batch size, and momentum.
   - Log the best hyperparameters and retrain the model using these parameters.

3. **Model Evaluation**

   - Evaluate the trained model on the CIFAR-10 test set.
   - Generate confusion matrix and per-class accuracy.

4. **DCGAN Implementation**

   - Implement a DCGAN to generate MNIST-like images.
   - Train the GAN and save generated images at each epoch.

## Requirements

- Python 3.6+
- PyTorch
- Optuna
- Matplotlib
- Scikit-learn
- Seaborn
- PySide6

## Installation

1. Clone this repository:

   ```sh
   git clone git@github.com:LeoLiao123/ncku-cvdl-hw.git
   ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Train the VGG19 model with SGD:

    ```sh
    python hw2/q1/train/train_sgd.py
    ```

2. Train the VGG19 model with Adam:

    ```sh
    python hw2/q1/train/train_adam.py
    ```

3. Evaluate the trained model:

    ```sh
    python hw2/q1/train/test_best.py
    ```

4. Train the DCGAN:

    ```sh
    python hw2/q2/train/train.py
    ```

5. Run the GUI application for Homework 2:

    ```sh
    python hw2/q1/main.py
    ```

    ```sh
    python hw2/q2/main.py
    ```

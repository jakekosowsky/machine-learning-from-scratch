"""Dense neural network for handwritten-digit classification."""

# Add import statements here
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

def neural_network(x_train, y_train, x_test, y_test):

  # Implement model
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(28,28)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(16, activation="tanh"))
  model.add(tf.keras.layers.Dense(16, activation="tanh"))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(16, activation="tanh"))
  model.add(tf.keras.layers.Dense(10))

  # Feel free to change this up, but leave it at first
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  # Fit and evaluate
  history = model.fit(x_train, y_train, batch_size=100, epochs=10)
  results = model.evaluate(x_test, y_test, batch_size=100)
  test_loss = results[0]
  test_acc = results[1]

  # Calculate predictions
  predictions = model.predict(x_test)

  return test_loss, test_acc, predictions

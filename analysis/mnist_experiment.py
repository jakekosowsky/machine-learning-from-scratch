# Other neural network code here:

# Load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(0, 25):
  fig.add_subplot(5, 5, i+1)
  plt.imshow(x_train[i, :])
plt.show()

x_train_scaled = x_train.astype(float)
x_test_scaled = x_test.astype(float)
for i in range(0, x_train.shape[0]):
  x_train_scaled[i, :] /= 255
for i in range(0, x_test.shape[0]):
  x_test_scaled[i, :] /= 255

test_loss, test_acc, predictions = neural_network(x_train_scaled, y_train, x_test_scaled, y_test)
print(test_loss, " | ", test_acc)

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

fig = plt.figure(figsize=(30, 30))
fig.subplots_adjust(hspace=1, wspace=0)
num_points = 20
for i in range(0, num_points):
  fig.add_subplot(num_points, 2, 2*i + 1)
  plt.xlabel("Number")
  plt.ylabel("Prediction")
  plt.bar(numbers, predictions[i])
  fig.add_subplot(num_points, 2, 2*(i+1))
  plt.imshow(x_test[i, :])
plt.show()

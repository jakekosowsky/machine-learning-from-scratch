"""Bootstrap aggregation implemented with individual decision trees."""

def out_of_bag(trees, pointsUsed, X_train, y_train):
  error = 0
  count = 0
  for i in range(0, X_train.shape[0]):
    numTrees = 0
    predict = 0
    for t in range(0, len(trees)):
      treePoints = pointsUsed[t]
      tree = trees[t]
      if i not in treePoints:
        numTrees += 1
        x = X_train[i].reshape((1, -1))
        y_class = tree.predict(x)
        predict += y_class
    if numTrees > 0:
      predict = np.sign(predict)
      y = y_train[i]
      count += 1
      if y != predict:
        error += 1
  out_of_bag_error = error/count

  return out_of_bag_error

def test(trees, X_test, y_test):
  ys = []
  for t in range(0, len(trees)):
    tree = trees[t]
    y_tree = tree.predict(X_test)
    ys.append(y_tree)
  error = 0
  for i in range(0, y_test.shape[0]):
    predict = 0
    y = y_test[i]
    for t in range(0, len(trees)):
      tree_result = ys[t]
      tree_y = tree_result[i]
      predict += tree_y
    predict = np.sign(predict)
    if predict != y:
      error += 1
  error /= y_test.shape[0]

  return error

def random_forest(X_train, y_train, X_test, y_test, num_bags, m):

  # Your code here, assign the proper values to out_of_bag_error and test_error:
  trees = []
  pointsUsed = []
  for t in range(0, num_bags):
    X = np.empty(X_train.shape)
    y = np.empty(y_train.shape)
    pointIndices = []
    for n in range(0, X_train.shape[0]):
      pointIndex = random.randint(0, X_train.shape[0]-1)
      if pointIndex not in pointIndices:
        pointIndices.append(pointIndex)
      X[n] = X_train[pointIndex]
      y[n] = y_train[pointIndex]
    pointsUsed.append(pointIndices)
    tree = DecisionTreeClassifier(max_features=m)
    tree.fit(X, y)
    trees.append(tree)
  out_of_bag_error = out_of_bag(trees, pointsUsed, X_train, y_train)
  test_error = test(trees, X_test, y_test)

  return out_of_bag_error, test_error

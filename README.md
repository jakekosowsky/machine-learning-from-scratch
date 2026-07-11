# Machine Learning From Scratch

Two classification systems organized to make their underlying mechanics visible: a dense neural network for handwritten-digit recognition and a bagged decision-tree ensemble with explicit bootstrap sampling and out-of-bag evaluation.

## Repository structure

```text
src/
  neural_network.py          model architecture and evaluation
  bagged_trees.py            bootstrap sampling, voting, and OOB error
analysis/
  mnist_experiment.py        preprocessing and digit-level diagnostics
  ensemble_experiment.py     ensemble-size and error comparisons
notebooks/
  ml_models.ipynb            full exploratory analysis
```

## Neural network

The digit classifier normalizes MNIST images, maps pixels through hidden dense layers, and compares predicted classes with observed labels. The analysis includes example images, accuracy measurement, and class-level prediction inspection.

## Bagged decision trees

The ensemble implementation trains individual decision trees on bootstrap samples, aggregates their predictions through majority voting, and evaluates observations that were left out of each tree's training sample. Keeping those steps explicit makes the relationship between sampling, ensemble size, test error, and out-of-bag error easy to inspect.

## Tools

Python · TensorFlow · scikit-learn · pandas · NumPy · SciPy · Matplotlib

# Machine Learning From Scratch

Implementations of two classification approaches: a neural network for handwritten-digit recognition and a bagged random-forest-style ensemble built from individual decision trees.

## What this demonstrates

- Neural-network construction and evaluation with TensorFlow/Keras
- Image normalization and multiclass prediction
- Bootstrap aggregation using decision trees
- Out-of-bag error estimation
- Model evaluation across ensemble sizes

## Repository structure

- `notebooks/ml_models.ipynb` — implementations, experiments, and visualizations
- `data/` — optional local datasets; MNIST is downloaded by TensorFlow

## Tools

Python, TensorFlow, scikit-learn, pandas, NumPy, SciPy, Matplotlib, and Jupyter.

## Note

The ensemble is implemented explicitly for educational clarity rather than by calling scikit-learn's `RandomForestClassifier`. This makes the bootstrap sampling, tree aggregation, and out-of-bag evaluation visible in code.

# thunder-extraction

> algorithms for feature extraction from spatio-temporal data

Source or feature extraction is the process of identifying spatial features of interest from data that varies over space and time. It can be either unsupervised or supervised, and is common in biological data analysis problems, like identifying neurons in calcium imaging data.

This package contains a collection of approaches for solving this problem. It defines a set of `algorithms` in the [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) style, each of which can be `fit` to data, and return a `model` that can be used to `transform` new data. Compatible with Python 2.7+ and 3.4+. Works well alongside [`thunder`](https://github.com/thunder-project/thunder) and supprts parallelization via [`spark`](https://github.com/apache/spark), but can be used as a standalone package on local [`numpy`](https://github.com/numpy/numpy) arrays.

## installation

```
pip install thunder-extraction
```

## examples

### algorithms

In this example we'll generate data and fit a model

```python
# generate data

from extraction.utils import make_gaussian
data = make_gaussian()

# fit a model

from extraction import NMF
model = NMF().fit(data)

# extract sources by transforming data

sources = model.transform(data)
```

## usage

Analysis starts by import and constructing an algorithm

```python
from extraction import NMF
algorithm = NMF(k=10)
```

Algorithms can be fit to data in the form of a `thunder` `images` object or an `x by y by z by t` `numpy` array

```python
model = algorithm.fit(data)
```

The model is a collection of identified features that can be used to extract temporal signals from new data

```python
signals = model.transform(data)
```

## api

### algorithms

All algorithms have the following methods

#### `algorithm.fit(data)`

Fits the algorithm to the data, which should be a collection of time-varying images. It can either be a [`thunder`](https://github.com/thunder-project/thunder) `images` object, or a `numpy` array with shape `t,x,y(,z)`.

### model

The result of fitting an `algorithm` is a `model`. Every `model` has the following properties and methods.

#### `model.regions`

The spatial regions identified during fitting.

#### `model.transform(data)`

Transform a new data set using the `model`, by averaging pixels within each of the `regions`. As with fitting, `data` can either be a [`thunder`](https://github.com/thunder-project/thunder) `images` object, or a `numpy` array with shape `t,x,y(,z)`. It will return a [`thunder`](https://github.com/thunder-project/thunder) `series` object.

## list of algorithms

Here are all the algorithms currently available.

#### `NMF(k=10, max_iterations=10, threshold=0.5, overlap=0.5)`

Perform local non-negative matrix factorization and thresholding to yield binary spatial regions.

- `k` number of components
- `max_iterations` maximum number of iterations to perform
- `threshold` percentile to use for thresholding
- `overlap` degree of overlap required for merging

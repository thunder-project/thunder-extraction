# thunder-extraction

[![Latest Version](https://img.shields.io/pypi/v/thunder-extraction.svg?style=flat-square)](https://pypi.python.org/pypi/thunder-extraction)
[![Build Status](https://img.shields.io/travis/thunder-project/thunder-extraction/master.svg?style=flat-square)](https://travis-ci.org/thunder-project/thunder-extraction) 

> algorithms for feature extraction from spatio-temporal data

Source or feature extraction is the process of identifying spatial features of interest from data that varies over space and time. It can be either unsupervised or supervised, and is common in biological data analysis problems, like identifying neurons in calcium imaging data.

This package contains a collection of approaches for solving this problem. It defines a set of `algorithms` in the [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) style, each of which can be `fit` to data, and return a `model` that can be used to `transform` new data. Compatible with Python 2.7+ and 3.4+. Works well alongside [`thunder`](https://github.com/thunder-project/thunder) and supprts parallelization via [`spark`](https://github.com/apache/spark), but can be used as a standalone package on local [`numpy`](https://github.com/numpy/numpy) arrays.

## installation

```
pip install thunder-extraction
```

## example

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

Algorithms can be fit to data in the form of a [`thunder`](https://github.com/thunder-project/thunder) `images` object or an `t,x,y(,z)` [`numpy`](https://github.com/numpy/numpy) array

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

#### `algorithm.fit(data, opts)`

Fits the algorithm to the data, which should be a collection of time-varying images. It can either be a [`thunder`](https://github.com/thunder-project/thunder) `images` object, or a [`numpy`](https://github.com/numpy/numpy) array with shape `t,x,y(,z)`.

### model

The result of fitting an `algorithm` is a `model`. Every `model` has the following properties and methods.

#### `model.regions`

The spatial regions identified during fitting.

#### `model.transform(data)`

Transform a new data set using the `model`, by averaging pixels within each of the `regions`. As with fitting, `data` can either be a [`thunder`](https://github.com/thunder-project/thunder) `images` object, or a `numpy` array with shape `t,x,y(,z)`. It will return a [`thunder`](https://github.com/thunder-project/thunder) `series` object, which can be converted to a [`numpy`](https://github.com/numpy/numpy) array by calling `toarray()`.

#### `model.merge(overlap=0.5, max_iter=2, k_nearest=10)`

Merge overlapping regions in the model, by greedily comparing nearby regions and merging those that are similar to one another. Only considers `k` nearest neighbors to speed up computation.

## list of algorithms

Here are all the algorithms currently available.

#### `NMF(k=5, max_iter=20, max_size='full', min_size=20, percentile=95, overlap=0.1)`

Local non-negative matrix factorization followed by thresholding to yield binary spatial regions. Applies factorization either to image blocks or to the entire image.

The algorithm takes the following parameters.

- `k` number of components to estimate per block
- `max_size` maximum size of each region
- `min_size` minimum size for each region
- `max_iter` maximum number of algorithm iterations
- `percentile` value for thresholding (higher means more thresholding)
- `overlap` value for determining whether to merge (higher means fewer merges) 

The fit method takes the following options.

- `block_size` a size in megabytes like `150` or a size in pixels like `(10,10)`, if `None` will use full image

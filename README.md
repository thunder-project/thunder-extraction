# thunder-extract

Algorithms for extracting features from spatial and temporal data. Includes a collection of `algorithms` that can be `fit` to data, all of which return a `model` that can be used to `transform` new data, in the `scikit-learn` style. Built on `numpy`, `scipy`, `sklearn`, and `skimage`. Works well alongside `thunder`, but can be used as a standalone module on local arrays.

# examples

### algorithms

running an algorithm

```python
from extract import NMF
model = NMF(params).fit(data)
```

transforming data

```python
result = model.transform(data)
```

### models

loading

```python
from extract import load
model = load('model.json')
result = model.transform(data)
```

saving

```python
model.save('model.json')
```

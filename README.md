# rime

Algorithms for feature extraction from spatiotemporal data. Includes methods for working with and rendering spatial sources, and saving them to/from disk.

# API reference

### algorithms

running an algorithm

```python
from rime.algorithms import LocalMax
sources = LocalMax(params).fit(data)
```

### sources

loading

```python
from rime import load
sources = load('sources.json')
```

saving

```python
sources.save('sources.json')
```

rendering

```python
im = sources.masks([x, y])
```

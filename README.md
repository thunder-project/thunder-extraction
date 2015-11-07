# rime

Algorithms for feature extraction from spatiotemporal data. Includes methods for working with and rendering spatial sources, and saving them to/from disk.

# API notes

## running an algorithm

```python
from rime.algorithms import LocalMax
model = LocalMax().fit(data)
```

## loading existing sources

```python
from rime import load
sources = load('sources.json')
```

## saving sources

```
sources.save('sources.json')
```

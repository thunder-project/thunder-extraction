import pytest
from numpy import arange, allclose, asarray, expand_dims
from scipy.spatial.distance import cdist

from extraction.utils import make_gaussian
from extraction import NMF

pytestmark = pytest.mark.usefixtures("eng")


def test_nmf_one(eng):
  data, series, truth = make_gaussian(n=1, noise=0.5, seed=42, engine=eng, withparams=True)
  algorithm = NMF()
  model = algorithm.fit(data, chunk_size=(100,200))
  assert model.regions.count == 1
  assert allclose(model.regions.center, truth.regions.center, 0.1)


def test_nmf_many(eng):
  data, series, truth = make_gaussian(n=5, noise=0.5, seed=42, engine=eng, withparams=True)
  algorithm = NMF()
  model = algorithm.fit(data, chunk_size=(100,200))
  assert model.regions.count == 5
  assert allclose(sum(cdist(model.regions.center, truth.regions.center) < 10), [1, 1, 1, 1,1])

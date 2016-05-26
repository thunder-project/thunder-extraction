import pytest
from numpy import arange, allclose, asarray, expand_dims
from regional import many
from thunder.images import fromarray

from extraction.model import ExtractionModel

pytestmark = pytest.mark.usefixtures("eng")


def test_model_construction():
    regions = many([[[0, 1], [0, 2]], [[0, 2], [0, 3]]])
    model = ExtractionModel(regions=regions)
    assert isinstance(model.regions, many)
    assert model.regions.count == 2


def test_model_transform(eng):
    regions = many([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
    model = ExtractionModel(regions=regions)
    im0 = [[0, 1], [1, 2]]
    im1 = [[3, 4], [5, 6]]
    im2 = [[7, 8], [9, 10]]
    data = fromarray([im0, im1, im2], engine=eng)
    transformed = model.transform(data)
    assert allclose(transformed.toarray(), [[0.5, 3.5, 7.5], [1.5, 5.5, 9.5]])


def test_model_transform_alt(eng):
    regions = many([[[0, 0], [0, 1]], [[1, 0]]])
    model = ExtractionModel(regions=regions)
    im0 = [[0, 1], [1, 2]]
    im1 = [[3, 4], [5, 6]]
    im2 = [[7, 8], [9, 10]]
    data = fromarray([im0, im1, im2], engine=eng)
    transformed = model.transform(data)
    assert allclose(transformed.toarray(), [[0.5, 3.5, 7.5], [1, 5, 9]])


def test_model_transform_single(eng):
    regions = many([[[0, 0], [0, 1]]])
    model = ExtractionModel(regions=regions)
    im0 = [[0, 1], [1, 2]]
    im1 = [[3, 4], [5, 6]]
    im2 = [[7, 8], [9, 10]]
    data = fromarray([im0, im1, im2], engine=eng)
    transformed = model.transform(data)
    assert allclose(transformed.toarray(), [0.5, 3.5, 7.5])
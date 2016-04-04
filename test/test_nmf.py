import pytest
from numpy import arange, allclose, asarray, expand_dims

from extraction.utils import make_gaussian
from extraction import NMF

pytestmark = pytest.mark.usefixtures("eng")

def test_nmf(eng):
  data = make_gaussian(engine=eng)

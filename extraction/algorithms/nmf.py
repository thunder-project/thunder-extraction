from ..model import ExtractionModel

class NMF(object):
  """
  Source extraction using non-negative matrix factorization
  """
  def __init__(self, k=10, max_iterations=10, threshold=0.5):
      self.k = k
      self.max_iterations = max_iterations
      self.threshold = threshold

  def fit(self, images):
      # do the computation
      # generate a set of regions
      # return a model
      pass
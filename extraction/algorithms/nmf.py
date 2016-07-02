from numpy import clip, inf, percentile, asarray, where, size, prod, unique, bincount
from scipy.ndimage import median_filter
from sklearn.decomposition import NMF as SKNMF
from skimage.measure import label
from skimage.morphology import remove_small_objects
import itertools

from regional import one, many
from ..utils import check_images
from ..model import ExtractionModel

class NMF(object):
  """
  Source extraction using local non-negative matrix factorization.
  """
  def __init__(self, k=5, max_iter=20, max_size='full', min_size=20, percentile=95, overlap=0.1):
      self.k = k
      self.max_iter = max_iter
      self.min_size = min_size
      self.max_size = max_size
      self.overlap = overlap
      self.percentile = percentile

  def fit(self, images, chunk_size=None, padding=None):
      images = check_images(images)
      chunk_size = chunk_size if chunk_size is not None else images.shape[1:]
      blocks = images.toblocks(chunk_size=chunk_size, padding=padding)
      sources = asarray(blocks.map_generic(self._get))

      # add offsets based on block coordinates
      for inds in itertools.product(*[range(d) for d in sources.shape]):
          offset = (asarray(inds) * asarray(blocks.blockshape)[1:])
          for source in sources[inds]:
              source.coordinates += offset
              if padding:
                leftpad = [blocks.padding[i + 1] if inds[i] != 0 else 0 for i in range(len(inds))]
                source.coordinates -= asarray(leftpad)
      
      # flatten list and create model
      flattened = list(itertools.chain.from_iterable(sources.flatten().tolist()))
      return ExtractionModel(many(flattened))

  def _get(self, block):
      """
      Perform NMF on a block to identify spatial regions.
      """
      dims = block.shape[1:]
      max_size = prod(dims) / 2 if self.max_size == 'full' else self.max_size

      # reshape to t x spatial dimensions
      data = block.reshape(block.shape[0], -1)

      # build and apply NMF model to block
      model = SKNMF(self.k, max_iter=self.max_iter)
      model.fit(clip(data, 0, inf))

      # reconstruct sources as spatial objects in one array
      components = model.components_.reshape((self.k,) + dims)

      # convert from basis functions into shape
      # by median filtering (optional), applying a percentile threshold,
      # finding connected components and removing small objects
      combined = []
      for component in components:
          tmp = component > percentile(component, self.percentile)
          labels, num = label(tmp, return_num=True)
          if num == 1:
            counts = bincount(labels.ravel())
            if counts[1] < self.min_size:
              continue
            else:
              regions = labels
          else:
            regions = remove_small_objects(labels, min_size=self.min_size)
          ids = unique(regions)
          ids = ids[ids > 0]
          for ii in ids:
              r = regions == ii
              r = median_filter(r, 2)
              coords = asarray(where(r)).T
              if (size(coords) > 0) and (size(coords) < max_size):
                  combined.append(one(coords))

      # merge overlapping sources
      if self.overlap is not None:

          # iterate over source pairs and find a pair to merge
          def merge(sources):
              for i1, s1 in enumerate(sources):
                  for i2, s2 in enumerate(sources[i1+1:]):
                      if s1.overlap(s2) > self.overlap:
                          return i1, i1 + 1 + i2
              return None

          # merge pairs until none left to merge
          pair = merge(combined)
          testing = True
          while testing:
              if pair is None:
                  testing = False
              else:
                  combined[pair[0]] = combined[pair[0]].merge(combined[pair[1]])
                  del combined[pair[1]]
                  pair = merge(combined)

      return combined
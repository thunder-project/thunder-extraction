from numpy import clip, inf, percentile, asarray, where, size, prod, unique, bincount, std, mean
from scipy.ndimage import median_filter
import cnmf
from skimage.measure import label
from skimage.morphology import remove_small_objects
import itertools

from regional import one, many
from ..utils import check_images
from ..model import ExtractionModel

class CNMFWRAP(object):
  """
  Source extraction using local non-negative matrix factorization.
  """
  def __init__(self, k=5, gSig=[4,4], merge_thresh=0.8, weight_thresh=0.0, p=2):
      self.k = k
      self.gSig = gSig
      self.merge_thresh = merge_thresh
      self.weight_thresh=weight_thresh
      self.p=p

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
      algorithm = cnmf.CNMF( k=self.k, gSig=self.gSig, merge_thresh=self.merge_thresh, p=self.p)

      model, temporaldata = algorithm.fit(block)
      regions=[]

      def convert(array):
        r,c = where(array > self.weight_thresh*array.max())#&&std(temporaldata())
        return one(zip(r,c))

      for i in range(model.shape[2]):
        region = convert(model[:,:,i])
        spike = temporaldata[i,:]
        if len(region.coordinates) > 0 and std(spike)>mean(spike)*.1:
          regions.append(region)
      #regions = [convert(model[:,:,i]) for i in range(model.shape[2])]
      return regions
     
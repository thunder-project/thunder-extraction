from regional import one, many
from numpy import mean, array

from .utils import check_images

class ExtractionModel(object):

    def __init__(self, regions):
        if isinstance(regions, list):
            self.regions = many(regions)
        elif isinstance(regions, many):
            self.regions = regions
        else:
            raise Exception("Input type not recognized, must be many regions")

    def transform(self, images):
        """
        Transform image data by averaging within regions.

        Will compute the average within 

        Parameters
        ----------
        images : array-like or thunder images object
          The image data to transform
        """
        images = check_images(images)
        
        ndims = len(images.dims)
        selections = []
        for region in self.regions:
            transformed = [[] for _ in range(ndims)]
            for indices in region.coordinates:
                for dim in range(ndims):
                    transformed[dim].append(indices[dim])
            selections.append([array(indices) for indices in transformed])

        def mean_by_indices(image):
            out = array([mean(image[indices]) for indices in selections]).reshape((1, -1))
            return out

        return images.map(mean_by_indices).toseries()
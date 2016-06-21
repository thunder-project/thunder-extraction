from regional import one, many
from numpy import mean, array
from copy import deepcopy

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

        Parameters
        ----------
        images : array-like or thunder images object
          The image data to transform
        """
        images = check_images(images)
        
        ndims = len(images.value_shape)
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

    def merge(self, overlap=0.5, max_iter=2):
        """
        Merge overlapping regions.

        Uses a greedy algorithm in which each region is merged with
        regions that have high overlap, concluding after one or more
        iterations are made over the full set.

        Parameters
        ----------
        overlap : float
            Minimal overlap for sources to be merged.

        max_iter : int
            Number of iterations.
        """
        def merge_once(initial):
            regions = []
            skip = []
            keep = []

            for ia, source in enumerate(initial):
                for ib, other in enumerate(initial):
                    if not ia == ib and source.overlap(other) > overlap:
                        source = source.merge(other)
                        if ib not in keep:
                            skip.append(ib)

                regions.append(source)
                keep.append(ia)

            return [region for ir, region in enumerate(regions) if ir not in skip]

        regions = merge_once(self.regions)

        for _ in range(max_iter-1):
            regions = merge_once(regions)

        return ExtractionModel(regions)

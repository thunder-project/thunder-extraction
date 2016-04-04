from numpy import asarray, ndarray
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from skimage.draw import circle

from thunder.images import fromarray, Images


def check_images(data):
    """
    Check and reformat input images if needed
    """
    if isinstance(data, ndarray):
        data = fromarray(data)
    
    if not isinstance(data, Images):
        data = fromarray(asarray(data))

    if len(data.shape) not in set([3, 4]):
        raise Exception('Number of image dimensions %s must be 2 or 3' % (len(data.shape)))

    return data

def make_gaussian(dims=(100, 200), centers=5, t=100, margin=35, sd=3, noise=0.1, npartitions=1, seed=None, withparams=True):

    from thunder.extraction.source import SourceModel

    random.seed(seed)

    if len(dims) != 2:
        raise Exception("Can only generate for two-dimensional sources.")

    if size(centers) == 1:
        n = centers
        xcenters = (dims[0] - margin) * random.random_sample(n) + margin/2
        ycenters = (dims[1] - margin) * random.random_sample(n) + margin/2
        centers = zip(xcenters, ycenters)
    else:
        centers = asarray(centers)
        n = len(centers)

    ts = [random.randn(t) for i in range(0, n)]
    ts = clip(asarray([gaussian_filter1d(vec, 5) for vec in ts]), 0, 1)
    for ii, tt in enumerate(ts):
        ts[ii] = (tt / tt.max()) * 2
    allframes = []
    for tt in range(0, t):
        frame = zeros(dims)
        for nn in range(0, n):
            base = zeros(dims)
            base[centers[nn][0], centers[nn][1]] = 1
            img = gaussian_filter(base, sd)
            img = img/max(img)
            frame += img * ts[nn][tt]
        frame += clip(random.randn(dims[0], dims[1]) * noise, 0, inf)
        allframes.append(frame)

    def pointToCircle(center, radius):
        rr, cc = circle(center[0], center[1], radius)
        return array(zip(rr, cc))

    r = round(sd * 1.5)
    sources = SourceModel([pointToCircle(c, r) for c in centers])

    data = fromList(allframes, npartitions).astype('float')
    if withparams is True:
        return data, ts, sources
    else:
        return data
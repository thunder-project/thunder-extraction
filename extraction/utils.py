from numpy import asarray, array, ndarray, random, clip, zeros, max, inf
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from skimage.draw import circle

from thunder.images import fromarray, fromlist, Images


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

def make_gaussian(shape=(100, 200), n=5, t=100, sd=3, noise=0.1, seed=None, engine=None, withparams=False):
    """
    Generate random gaussian source data.

    Uses a spatial mixture gaussians with time-varying amplitudes.

    Parameters
    ----------
    shape : tuple, optional, default = (100,200)
        Shape of data.

    n : int, optional, default = 5
        Number of sources.

    t : int, optional, default = 100
        Number of time points.

    sd : float, optional, default = 3.0
        Standard deviation of gaussians.

    noise : float, optional, default = 0.1
        Random noise to add to result.

    seed : int, optional, default = None
        Random seed.

    engine : computational backend, optional, default = None
        Can be None (for local) or a SparkContext (for spark)

    withparams : bool, optionla, default = False
        If True, returns generating parameters along with data.
    """
    from .model import ExtractionModel

    random.seed(seed)

    margin = [shape[0] * 0.1, shape[1] * 0.1]
    xcenters = (shape[0] - margin[0]) * random.random_sample(n) + margin[0]/2
    ycenters = (shape[1] - margin[1]) * random.random_sample(n) + margin[1]/2
    centers = list(zip(xcenters, ycenters))

    series = [random.randn(t) for i in range(0, n)]
    series = clip(asarray([gaussian_filter1d(vec, 5) for vec in series]), 0, 1)
    for ii, tt in enumerate(series):
        series[ii] = (tt / (tt.max() + 0.01)) * 2

    frames = []
    for tt in range(t):
        frame = zeros(shape)
        for nn in range(n):
            base = zeros(shape)
            base[int(centers[nn][0]), int(centers[nn][1])] = 1
            img = gaussian_filter(base, sd)
            img = img/max(img)
            frame += img * series[nn][tt]
        frame += clip(random.randn(shape[0], shape[1]) * noise, 0, inf)
        frames.append(frame)

    def point_to_circle(center, radius):
        rr, cc = circle(center[0], center[1], radius)
        return array(list(zip(rr, cc)))

    r = round(sd * 1.5)

    model = ExtractionModel([point_to_circle(c, r) for c in centers])

    data = fromlist(frames, engine=engine).astype('float')
    if withparams is True:
        return data, series, model
    else:
        return data
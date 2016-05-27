# generate data

from extraction.utils import make_gaussian
data = make_gaussian(noise=0.5)

# fit a model

from extraction import NMF
model = NMF().fit(data, chunk_size=(100,200))

# show estimated sources

import matplotlib.pyplot as plt
from showit import image

image(model.regions.mask((100, 200), fill=None, stroke='deeppink', base=data.mean().toarray() / 2))
plt.show()
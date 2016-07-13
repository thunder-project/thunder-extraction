# generate data

from extraction.utils import make_gaussian
data = make_gaussian(noise=0.5)

# fit an nmf model

from extraction import NMF
nmfmodel = NMF().fit(data, chunk_size=(100,200))

# OR

#fit a cnmf model
from extraction import cnmf
cnmfmodel = CNMF().fit(data, chunk_size=(100,200))

# show estimated sources

import matplotlib.pyplot as plt
from showit import image

image(nmfmodel.regions.mask((100, 200), fill=None, stroke='deeppink', base=data.mean().toarray() / 2))
plt.show()

image(cnmfmodel.regions.mask((100, 200), fill=None, stroke='deeppink', base=data.mean().toarray() / 2))
plt.show()
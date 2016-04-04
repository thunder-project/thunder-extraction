# generate data

from extraction import make_data
data = make_data()

# fit a model

from extraction import NMF
model = NMF(k=10).fit(data)

# extract sources by transforming data

sources = model.transform(data)
# generate data

from pyspark import SparkContext
sc = SparkContext()

from extraction.utils import make_gaussian
data = make_gaussian(engine=sc)

# fit a model

from extraction import NMF
model = NMF().fit(data)

# extract sources by transforming data

sources = model.transform(data)

print model.regions.count
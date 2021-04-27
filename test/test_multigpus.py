import biodatasets
from biotransformers import BioTransformers

data = biodatasets.load_dataset('swissProt')
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

tr = BioTransformers("protbert",device="cuda:0",multi_gpu=True)
tr.compute_embeddings(X[:100],batch_size=16)
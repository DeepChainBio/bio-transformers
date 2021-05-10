# type: ignore
import biodatasets
from biotransformers import BioTransformers

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# tr = BioTransformers("esm1b_t33_650M_UR50S",device="cuda",multi_gpu=True)
tr = BioTransformers(device="cuda", multi_gpu=True)
# tr.compute_embeddings(X[:2000], batch_size=48)  # noqa
tr.train_masked(X[:2000], batch_size=8, epochs=30)

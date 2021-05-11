# type: ignore
import biodatasets
from biotransformers import BioTransformers
import numpy as np

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# Train on small sequence
length = np.array(list(map(len, X))) < 500

# tr = BioTransformers("esm1b_t33_650M_UR50S",device="cuda",multi_gpu=True)
tr = BioTransformers("esm1_t12_85M_UR50S", device="cuda", multi_gpu=True)
# tr.compute_embeddings(X[:2000], batch_size=48)  # noqa
tr.train_masked(X[length], batch_size=8, epochs=20)

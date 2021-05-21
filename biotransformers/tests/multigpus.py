# type: ignore
import biodatasets
import numpy as np
from biotransformers import BioTransformers

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# Train on small sequence
length = np.array(list(map(len, X))) < 100
print("Number of proteins {}".format(sum(length)))

tr = BioTransformers("esm1_t6_43M_UR50S", device="cuda", multi_gpu=True)
tr.train_masked(X[length][10000], batch_size=32, epochs=2)
print(tr.model)

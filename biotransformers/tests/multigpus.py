# type: ignore
import biodatasets
import numpy as np
from biotransformers import BioTransformers

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# Train on small sequence
length = np.array(list(map(len, X))) < 200
print("Number of proteins {}".format(sum(length)))

tr = BioTransformers("esm1_t6_43M_UR50S", device="cuda", multi_gpu=True)
tr.train_masked(X[length], batch_size=16, epochs=20)

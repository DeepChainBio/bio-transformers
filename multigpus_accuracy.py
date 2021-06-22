# type: ignore
import biodatasets
import numpy as np
import ray

from biotransformers import BioTransformers

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# Train sequence with length less than 200 AA
length = np.array(list(map(len, X))) < 200
train_seq = X[length][15000:16000]
print("Test on {} of protein sequence".format(len(train_seq)))

ray.init()
bio_trans = BioTransformers("protbert", num_gpus=2)

acc_before = bio_trans.compute_accuracy(train_seq, batch_size=128)
print(f"acc before {acc_before}")
# bio_trans.load_model(
#    "/home/a.delfosse/bio-transformers/logs/finetune_masked/version_2/esm1_t6_43M_UR50S_finetuned.pt"
# )
# acc_after = bio_trans.compute_accuracy(train_seq, batch_size=64)
# print(f"acc after {acc_after}")

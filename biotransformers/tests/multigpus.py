# type: ignore
import biodatasets
import numpy as np
from biotransformers import BioTransformers

data = biodatasets.load_dataset("swissProt")
X, y = data.to_npy_arrays(input_names=["sequence"])
X = X[0]

# Train on small sequence
length = np.array(list(map(len, X))) < 200
train_seq = X[length][:1000]
# print("Number of proteins {}".format(sum(length)))


def print_accuracy(acc_score):
    print("ACCURACY")
    print("Accuracy score : {}".format(acc_score))
    print("--------------")


bio_trans = BioTransformers("esm1_t6_43M_UR50S", device="cuda", multi_gpu=False)

acc_before = bio_trans.compute_accuracy(
    "/home/a.delfosse/data_swissprot.fasta", batch_size=16
)
print_accuracy(acc_before)

# bio_trans.train_masked(
#    train_seq,
#    toks_per_batch=500,
#    epochs=2,
#    accelerator="dp",
# )

# bio_trans.load_model("esm1_t6_43M_UR50S_finetuned.pt")

acc_after = bio_trans.compute_accuracy(train_seq, batch_size=16)
print_accuracy(acc_after)

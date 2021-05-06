import collections
import datetime
import functools
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple
import argparse
import numpy as np
import torch
import pandas as pd
import esm
from tqdm import tqdm
from esm.data import Alphabet, BatchConverter
from torch.nn import functional as F  # noqa: N812 pylint: disable=wrong-import-order
from torch.utils.data import DataLoader 
from torch import nn

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

NATURAL_AAS_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Train ESM.")
    parser.add_argument(
        "--model_name", type=str, default="esm1b_t33_650M_UR50S", help="ESM model name."
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=2, help="Minibatch size for training."
    )
    parser.add_argument(
        "--acc_batch_size", type=int, default=2048, help="Accumulated batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1.0e-5, help="learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument(
        "--save_period", type=int, default=50, help="Save checkpoint every X epochs."
    )
    parser.add_argument("--filter_len", dest="filter_len", action="store_true")
    parser.add_argument("--no-filter_len", dest="filter_len", action="store_false")
    parser.set_defaults(filter_len=True)
    parser.add_argument(
        "--output_dir",
        help="Path for output.",
        type=str,
        default="out",
    )
    parser.add_argument(
        "--cache_dir",
        help="Path for cache.",
        type=str,
        default="cache",
    )
    parser.add_argument(
        "--num_layers_to_freeze", type=int, default=0, help="Number of layers to freeze."
    )
    parser.add_argument("--name", type=str, default="ESM_Train", help="Name of the experiment.")
    parser.add_argument("--neptune", dest="neptune", action="store_true")
    parser.add_argument("--no-neptune", dest="neptune", action="store_false")
    parser.set_defaults(neptune=True)
    parser.add_argument(
        "--masking_ratio", type=float, default=0.025, help="Ratio of sequences to be masked."
    )
    parser.add_argument(
        "--masking_prob", type=float, default=0.8, help="Probability of selected AA to be masked."
    )
    parser.add_argument(
        "--random_token_prob",
        type=float,
        default=0.1,
        help="Probability of selected AA to be randomly replaced.",
    )
    args = parser.parse_args()
    return args

def worker_init_fn(worker_id: int):
    """Set numpy random seed for each worker.

    https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359

    Args:
        worker_id: unique id for each worker
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def mask_seq(
    seq: str,
    tokens: torch.Tensor,
    prepend_bos: bool,
    mask_idx: int,
    pad_idx: int,
    masking_ratio: float,
    masking_prob: float,
    random_token_prob: float,
    random_token_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask one sequence randomly.

    Args:
        seq: string of the sequence.
        tokens: tokens corresponding to the sequence, length can be longer than the seq.
        prepend_bos: if tokenizer adds <bos> token
        mask_idx: index of the mask token
        pad_idx: index of the padding token
        masking_ratio: ratio of tokens to be masked.
        masking_prob: probability that the chose token is replaced with a mask token.
        random_token_prob: probability that the chose token is replaced with a random token.
        random_token_indices: list of token indices that random replacement selects from.

    Returns:
        tokens: masked tokens
        targets: same length as tokens
    """
    # init
    seq_len = len(seq)
    mask_num = int(np.ceil(seq_len * masking_ratio))
    targets = tokens.detach().clone()
    # sample indices
    mask_indices = sorted(np.random.choice(seq_len, mask_num, replace=False) + int(prepend_bos))
    # mask tokens
    for idx in mask_indices:
        rand = np.random.random()

        # replace with mask
        if rand < masking_prob:
            tokens[idx] = mask_idx

        # replace with random token
        elif rand < masking_prob + random_token_prob:
            tokens[idx] = np.random.choice(random_token_indices, 1)[0]

    # generate targets
    non_mask_indices = [i for i in range(seq_len) if i not in mask_indices]
    targets[non_mask_indices] = pad_idx
    return tokens, targets

def collate_fn(
    samples: Sequence[Tuple[str, str]],
    tokenizer: BatchConverter,
    alphabet: Alphabet,
    masking_ratio: float,
    masking_prob: float,
    random_token_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function to mask tokens.

    Args:
        samples: a sequences of (label, seq).
        tokenizer: facebook tokenizer, that accepts sequences of (label, seq_str)
            and outputs (labels, seq_strs, tokens).
        alphabet: facebook alphabet.
        masking_ratio: ratio of tokens to be masked.
        masking_prob: probability that the chose token is replaced with a mask token.
        random_token_prob: probability that the chose token is replaced with a random token.

    Returns:
        tokens: model input
        targets: model target
        mask_indices: indices of masked tokens
    """
    random_token_indices = [alphabet.tok_to_idx[aa] for aa in NATURAL_AAS_LIST]
    _, seqs, tokens = tokenizer(samples)

    tokens_list, targets_list = [], []
    for i, seq in enumerate(seqs):
        tokens_i, targets_i = mask_seq(
            seq=seq,
            tokens=tokens[i, :],
            prepend_bos=alphabet.prepend_bos,
            mask_idx=alphabet.mask_idx,
            pad_idx=alphabet.padding_idx,
            masking_ratio=masking_ratio,
            masking_prob=masking_prob,
            random_token_prob=random_token_prob,
            random_token_indices=random_token_indices,
        )
        tokens_list.append(tokens_i)
        targets_list.append(targets_i)

    tokens = torch.stack(tokens_list)
    targets = torch.stack(targets_list)

    return tokens, targets

def create_dataset(
    sequences: List[str],
    alphabet: Alphabet,
    filter_len: bool,
    batch_size: int,
    masking_ratio: float,
    masking_prob: float,
    random_token_prob: float,
    num_workers: int = 0,
) -> DataLoader:
    """Create the PyTorch Dataset.

    Args:
        filenames: list of fasta files.
        alphabet: facebook alphabet.
        filter_len: whether filter data wrt len.
        batch_size: num samples per batchs
        num_workers: num of parallel data samplers
        masking_ratio: ratio of tokens to be masked.
        masking_prob: probability that the chose token is replaced with a mask token.
        random_token_prob: probability that the chose token is replaced with a random token.

    Returns:
        torch DataLoader
    """
    sequences = enumerate(sequences)
    sequences = list(sequences)
    logging.info("Samples: %d", len(sequences))

    tokenizer = alphabet.get_batch_converter()

    loader = DataLoader(
        sequences,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=functools.partial(
            collate_fn,
            tokenizer=tokenizer,
            alphabet=alphabet,
            masking_ratio=masking_ratio,
            masking_prob=masking_prob,
            random_token_prob=random_token_prob,
        ),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    return loader

def accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int, reduction: str) -> float:
    """Calculate accuracy for multi-masking.

    Args:
        logits: shape = (batch, len_tokens, len_vocab)
        targets: shape = (batch, len_tokens)
        pad_idx: index to be ignored for targets
        reduction: sum or mean

    Returns:
        accuracy

    Raises:
        ValueError: if reduction is unknown.
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError(f"Reduction {reduction} is unknown. Supported are mean and sum.")

    preds = torch.argmax(logits, dim=-1)  # (batch, len_tokens)
    masked_tokens = targets.ne(pad_idx)

    masked_preds = torch.masked_select(preds, masked_tokens)
    masked_targets = torch.masked_select(targets, masked_tokens)

    matching = masked_preds == masked_targets
    acc = matching.sum().item()
    if reduction == "mean":
        sample_size = masked_tokens.int().sum().item()
        acc /= sample_size
    return acc

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate.

    Args:
        optimizer: optimizer object.

    Returns:
        learning rate.

    Raises:
        ValueError: if no param_groups found.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    raise ValueError


def main():  # noqa: CCR001
    """Launch training for ESM model."""
    # parse arguments
    args = parse_args()

    # add timestamp to output dir
    args.output_dir = args.output_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # get max_seq_len
    # load and filter data
    path_train = "/home/a.delfosse/project/protein_trainer/data/train_finetune_sequence.csv"
    path_test = "/home/a.delfosse/project/protein_trainer/data/test_finetune_sequence.csv"
    seqs_train = pd.read_csv(path_train)['sequence'].tolist()
    seqs_valid = pd.read_csv(path_train)['sequence'].tolist()

    # build model
    # TODO need to figure out why we need 2 here
    max_seq_len = max(len(seq) for seq in seqs_train)
    if seqs_valid:
        max_seq_len = max(max_seq_len, max(len(seq) for seq in seqs_valid))
    max_seq_len += 2

    grad_accumulation = args.acc_batch_size // args.batch_size

    # transformer
    device_ = torch.device("cuda")
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm1_t6_43M_UR50S")
    model = nn.DataParallel(model)
    model = model.to(device_)
    num_params = sum(np.prod(p.shape) for p in model.parameters())
    logging.info("Number of parameters: %d", num_params)

    # dataset
    training_loader = create_dataset(
        sequences=seqs_train,
        alphabet=alphabet,
        filter_len=args.filter_len,
        batch_size=args.batch_size,
        num_workers=4,
        masking_ratio=args.masking_ratio,
        masking_prob=args.masking_prob,
        random_token_prob=args.random_token_prob,
    )
    valid_loader = create_dataset(
        sequences=seqs_valid,
        alphabet=alphabet,
        filter_len=args.filter_len,
        batch_size=args.batch_size,
        num_workers=4,
        masking_ratio=args.masking_ratio,
        masking_prob=args.masking_prob,
        random_token_prob=args.random_token_prob,
    )

    # create optimizer
    warmup_updates = 10
    warmup_init_lr = min(1e-7, args.lr)

    parameters = [p for p in model.parameters() if p.requires_grad]
    logging.info("Trainable weights: %d", sum(np.prod(p.shape) for p in parameters))
    optimizer = torch.optim.Adam(params=parameters, lr=warmup_init_lr)
    optimizer.zero_grad()

    warmup_end_lr = args.lr
    lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
    decay_factor = warmup_end_lr * warmup_updates ** 0.5

    def lr_update(num_updates: int) -> float:
        """InverseSquareRootSchedule.

        https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py#L32

        Args:
            num_updates: number of batches

        Returns:
            learning rate multiplicate factor
        """
        if num_updates < warmup_updates:
            lr = warmup_init_lr + num_updates * lr_step
        else:
            lr = decay_factor * num_updates ** -0.5
        if warmup_init_lr > 0:
            return lr / warmup_init_lr
        return 0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: lr_update(num_updates=x)
    )

    #model.train()
    losses_train = 0.0
    step_counter = 0
    best_valid_acc = 0
    counter_train = collections.Counter()
    for epoch in range(args.epochs):
        # reset numpy random seed
        np.random.seed()
        for sample in tqdm(training_loader):
            # batch_tokens and targets have both shape (batch, len_tokens)
            batch_tokens, targets = [x.to(device_) for x in sample]

            # forward + loss computation
            with torch.cuda.amp.autocast():
                results = model(batch_tokens)
                masked_tokens = targets.ne(alphabet.padding_idx)
                sample_size = masked_tokens.int().sum()
                logits = results["logits"]
                print(logits.view(-1, logits.size(-1)).shape)
                print(targets.view(-1).shape)
                loss = (
                    F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        reduction="sum",
                        ignore_index=alphabet.padding_idx,
                    )
                    / sample_size
                    / grad_accumulation
                )  # normalize

            # optimization step
            loss.backward()
            step_counter += 1

            losses_train += loss.item()
            counter_train.update(
                {
                    "accuracy": accuracy(
                        logits, targets, pad_idx=alphabet.padding_idx, reduction="sum"
                    ),
                    "sample_size": sample_size.item(),
                }
            )

            if (step_counter + 1) % grad_accumulation == 0:
                logging.info("optimizer step")
                optimizer.step()
                optimizer.zero_grad()

                # logging
                metrics_train = {
                    "loss": losses_train,
                    "lr": get_lr(optimizer=optimizer),
                    "accuracy": counter_train["accuracy"] / counter_train["sample_size"],
                    "sample_size": float(counter_train["sample_size"]) / args.acc_batch_size,
                }

                logging.info("Train | Step %d | %s", step_counter, metrics_train)
                losses_train, counter_train = 0.0, collections.Counter()

                lr_scheduler.step()

        with torch.no_grad():
            num_seqs, num_samples, valid_loss, valid_acc = 0.0, 0.0, 0.0, 0.0
            for sample in valid_loader:
                batch_tokens, targets = [x.to(device_) for x in sample]
                with torch.cuda.amp.autocast():
                    results = model(batch_tokens)
                    masked_tokens = targets.ne(alphabet.padding_idx)
                    sample_size = masked_tokens.int().sum()
                    logits = results["logits"]
                num_samples += sample_size.item()
                num_seqs += batch_tokens.shape[0]
                valid_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="sum",
                    ignore_index=alphabet.padding_idx,
                ).item()
                valid_acc += accuracy(
                    logits, targets, pad_idx=alphabet.padding_idx, reduction="sum"
                )

            valid_acc = valid_acc / num_samples
            valid_loss = valid_loss / num_samples

            metrics_eval = {
                "accuracy": valid_acc,
                "loss": valid_loss,
                "num_samples": num_samples / num_seqs,
            }
            logging.info("Eval | Epoch %d | %s", epoch, metrics_eval)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                filename = Path(args.output_dir) / "esm1b_best.pt"
                logging.info("The best checkpoint %s saved at epoch %d", filename, epoch)
                torch.save(model.module.state_dict(), filename)
                # if args.neptune:
                #     try:
                #         neptune.log_artifact(str(Path.cwd() / filename))
                #         logging.info("The best checkpoint %s uploaded", filename)
                #     # pylint: disable=broad-except
                #     except Exception as err:  # noqa: W0703
                #         logging.info("Error %s", err)
                #         logging.info("The best checkpoint %s not uploaded", filename)

        # overwrite the same file
        if epoch % args.save_period == 0 or epoch == args.epochs - 1:
            filename = Path(args.output_dir) / f"{esm_model.model_name_}_{epoch}.pt"
            logging.info("Checkpoint %s saved", filename)
            torch.save(model.module.state_dict(), filename)
            # if args.neptune:
            #     try:
            #         neptune.log_artifact(str(Path.cwd() / filename))
            #         logging.info("Checkpoint %s uploaded", filename)
            #     # pylint: disable=broad-except
            #     except Exception as err:  # noqa: W0703
            #         logging.info("Error %s", err)
            #         logging.info("Checkpoint %s not uploaded", filename)

if __name__ == '__main__':
    main()
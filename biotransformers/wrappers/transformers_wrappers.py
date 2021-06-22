"""
This script defines a parent class for transformers, for which child classes which are
specific to a given transformers implementation can inherit.
It allows to derive probabilities, embeddings and log-likelihoods based on inputs
sequences, and displays some properties of the transformer model.
"""
import math
import os
from copy import deepcopy
from os.path import join
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import ray
import torch
import torch.tensor
from biotransformers.utils.constant import NATURAL_AAS_LIST
from biotransformers.utils.logger import logger  # noqa
from biotransformers.utils.tqdm_utils import ProgressBar
from biotransformers.utils.utils import (
    _check_memory_embeddings,
    _check_memory_logits,
    _check_sequence,
    _check_tokens_list,
    get_logs_version,
    load_fasta,
)
from biotransformers.wrappers.language_model import LanguageModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from ..lightning_utils.data import BioDataModule
from ..lightning_utils.models import LightningModule

log = logger("transformers_wrapper")


class TransformersWrapper:
    """
    Abstract class that uses pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(
        self,
        model_dir: str,
        language_model_cls: Type[LanguageModel],
        num_gpus: int = 0,
    ):
        """Initialize Transformers wrapper

        TODO : os.environ["CUDA_VISIBLE_DEVICES"]="0" or export CUDA_VISIBLE_DEVICES="0,1"

        Args:
            model_dir: name directory of the pretrained model
            num_gpus: number of gpus to use. If set to 0, it uses the cpu.
        """
        self._model_dir = model_dir
        self._num_gpus = num_gpus
        if num_gpus >= 1:
            assert torch.cuda.is_available(), "Cuda is not available."
            assert torch.cuda.device_count() >= num_gpus, "Not enough available GPUs."
        if num_gpus <= 1:
            device = "cpu" if num_gpus == 0 else "cuda"
            self._language_model = language_model_cls(model_dir=model_dir, device=device)
            self._multi_gpus = False
        else:
            self._language_model = language_model_cls(model_dir=model_dir, device="cpu")
            self._ray_cls = ray.remote(num_cpus=4, num_gpus=1)(language_model_cls)
            self._workers = [
                self._ray_cls.remote(model_dir=model_dir, device="cuda:0") for _ in range(num_gpus)
            ]
            self._multi_gpus = True

    def get_vocabulary_mask(self, tokens_list: List[str]) -> np.ndarray:
        """Returns a mask ove the model tokens."""
        # Compute a mask over vocabulary to decide which value to keep or not
        vocabulary_mask = np.array(
            [
                1.0 if token in tokens_list else 0.0
                for token in self._language_model.model_vocabulary
            ]
        )
        return vocabulary_mask

    def _get_num_batch_iter(self, model_inputs: Dict[str, Any], batch_size: int) -> int:
        """
        Get the number of batches when spliting model_inputs into chunks of size batch_size.
        """
        num_of_sequences = model_inputs["input_ids"].shape[0]
        num_batch_iter = int(np.ceil(num_of_sequences / batch_size))
        return num_batch_iter

    def _generate_chunks(
        self, model_inputs: Dict[str, Any], batch_size: int
    ) -> Generator[Dict[str, Iterable], None, None]:
        """Yield a dictionnary of tensor"""
        num_of_sequences = model_inputs["input_ids"].shape[0]
        for i in range(0, num_of_sequences, batch_size):
            batch_sequence = {
                key: value[i : (i + batch_size)] for key, value in model_inputs.items()
            }
            yield batch_sequence

    def _repeat_and_mask_inputs(
        self, model_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[List]]:
        """Create new tensor by masking each token and repeating sequence

        Args:
            model_inputs: shape -> (num_seqs, max_seq_len)

        Returns:
            model_inputs: shape -> (sum_tokens, max_seq_len)
            masked_ids_list: len -> (num_seqs)
        """
        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        mask_ids = []
        for sequence, binary_mask, zeros in zip(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["token_type_ids"],
        ):
            mask_id = []
            mask_token = self._language_model.mask_token
            does_end_token_exist = self._language_model.does_end_token_exist

            for i in range(1, sum(binary_mask) - does_end_token_exist * 1):
                mask_sequence = torch.tensor(
                    sequence[:i].tolist()
                    + [self._language_model.token_to_id(mask_token)]
                    + sequence[i + 1 :].tolist(),
                    dtype=torch.int64,
                )
                new_input_ids.append(mask_sequence)
                new_attention_mask.append(binary_mask)
                new_token_type_ids.append(zeros)
                mask_id.append(i)
            mask_ids.append(mask_id)

        model_inputs_out = deepcopy(model_inputs)
        model_inputs_out["input_ids"] = torch.stack(new_input_ids)
        model_inputs_out["attention_mask"] = torch.stack(new_attention_mask)
        model_inputs_out["token_type_ids"] = torch.stack(new_token_type_ids)

        return model_inputs_out, mask_ids

    def _gather_masked_outputs(
        self, model_outputs: torch.Tensor, masked_ids_list: List[List]
    ) -> torch.Tensor:
        """Gather all the masked outputs to get original tensor shape

        Args:
            model_outputs (torch.Tensor): shape -> (sum_tokens, max_seq_len, vocab_size)
            masked_ids_list (List[List]) : len -> (num_seqs)

        Returns:
            model_outputs (torch.Tensor): shape -> (num_seqs, max_seq_len, vocab_size)
        """
        max_length = model_outputs.shape[1]
        inf_tensor = -float("Inf") * torch.ones([1, model_outputs.shape[2]], dtype=torch.float32)
        sequences_list = []
        start_id = 0
        for mask_id in masked_ids_list:
            end_id = start_id + len(mask_id)
            sequence = torch.cat(
                (
                    inf_tensor,
                    model_outputs[range(start_id, end_id), mask_id],
                    inf_tensor.repeat(max_length - len(mask_id) - 1, 1),
                ),
                0,
            )
            sequences_list.append(sequence)
            start_id = end_id
        return torch.stack(sequences_list)

    def _model_evaluation(
        self, model_inputs: Dict[str, torch.tensor], batch_size: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and embeddings

        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        Args:
            model_inputs (Dict[str, torch.tensor]): [description]
            batch_size (int): [description]

        Returns:
            Tuple[torch.tensor, torch.tensor]:
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        silent = kwargs.get("silent", False)
        num_inputs = model_inputs["input_ids"].shape[0]
        n_updates = math.ceil(num_inputs / batch_size)
        if self._multi_gpus:
            jobs = []
            # Initialize logits and embeddings before looping over batches
            logits = torch.Tensor()  # [num_seqs, max_len_seqs+1, vocab_size]
            embeddings = torch.Tensor()  # [num_seqs, max_len_seqs+1, embedding_size]
            pb = ProgressBar(n_updates)
            actor = pb.actor
            for i, batch_inputs in enumerate(
                self._generate_chunks(model_inputs, math.ceil(num_inputs / self._num_gpus))
            ):
                # Split large batch into smaller batches, when per GPU worker
                # Send tqdm progress bar
                jobs.append(
                    self._workers[i].model_pass.remote(batch_inputs, batch_size, silent, actor)
                )
            pb.print_until_done()
            # Launch parallel execution in background
            outs = ray.get(jobs)
            # Gather the result
            for batch_logits, batch_embeddings in outs:
                embeddings = torch.cat((embeddings, batch_embeddings), dim=0)
                logits = torch.cat((logits, batch_logits), dim=0)
        else:
            logits, embeddings = self._language_model.model_pass(model_inputs, batch_size, silent)

        return logits, embeddings

    def _compute_logits(
        self, model_inputs: Dict[str, torch.Tensor], batch_size: int, pass_mode: str, **kwargs
    ) -> torch.Tensor:
        """Intermediate function to compute logits

        Args:
            model_inputs[str] (torch.Tensor): shape -> (num_seqs, max_seq_len)
            batch_size (int)
            pass_mode (str)

        Returns:
            logits (torch.Tensor): shape -> (num_seqs, max_seq_len, vocab_size)
        """
        if pass_mode == "masked":
            model_inputs, masked_ids_list = self._repeat_and_mask_inputs(model_inputs)
            logits, _ = self._model_evaluation(model_inputs, batch_size=batch_size, **kwargs)
            logits = self._gather_masked_outputs(logits, masked_ids_list)
        elif pass_mode == "forward":
            logits, _ = self._model_evaluation(model_inputs, batch_size=batch_size, **kwargs)
        return logits

    def compute_logits(
        self,
        sequences: Union[List[str], str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> List[np.ndarray]:
        """Function that computes the logits from sequences.

        It returns a list of logits arrays for each sequence.

        Args:
            sequences_list: List of sequences
            batch_size: number of sequences to consider for the forward pass
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            silent: whether to print progress bar in console

        Returns:
            List[np.ndarray]: logits in np.ndarray format
        """

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)
        _check_sequence(sequences, self._model_dir, 1024)
        _check_memory_logits(sequences, self._language_model.vocab_size, pass_mode)

        # Perform inference in model to compute the logits
        inputs = self._language_model.process_sequences_and_tokens(sequences)
        labels = torch.unsqueeze(inputs["input_ids"], dim=-1)
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        # Remove padded logits
        lengths = [len(sequence) for sequence in sequences]
        logits = [logit[:length, :] for logit, length in zip(list(logits), lengths)]
        labels = [label[:length, :] for label, length in zip(list(labels), lengths)]
        # Keep only corresponding to amino acids that are in the sequence
        logits = [
            torch.gather(logit, dim=-1, index=label).numpy() for logit, label in zip(logits, labels)
        ]
        # List of arrays of shape (seq_length, 1)
        return logits

    def compute_probabilities(
        self,
        sequences: Union[List[str], str],
        batch_size: int = 1,
        tokens_list: List[str] = None,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> List[Dict[int, Dict[str, float]]]:
        """Function that computes the probabilities over amino-acids from sequences.

        It takes as inputs a list of sequences and returns a list of dictionaries.
        Each dictionary contains the probabilities over the natural amino-acids for each
        position in the sequence. The keys represent the positions (indexed
        starting with 0) and the values are dictionaries of probabilities over
        the natural amino-acids for this position.

        In these dictionaries, the keys are the amino-acids and the value
        the corresponding probabilities.

        Both ProtBert and ESM models have more tokens than the 20 natural amino-acids
        (for instance MASK or PAD tokens). It might not be of interest to take these
        tokens into account when computing probabilities or log-likelihood. By default
        we remove them and compute probabilities only over the 20 natural amino-acids.
        This behavior can be overridden through the tokens_list argument that enable
        the user to choose the tokens to consider when computing probabilities.

        Args:
            sequences: List of sequences
            batch_size: number of sequences to consider for the forward pass
            tokens_list: List of tokens to consider
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            silent : display or not progress bar

        Returns:
            List[Dict[int, Dict[str, float]]]: dictionaries of probabilities per seq
        """
        if isinstance(sequences, str):
            sequences = load_fasta(sequences)
        tokens_list = NATURAL_AAS_LIST if tokens_list is None else tokens_list
        _check_sequence(sequences, self._model_dir, 1024)
        _check_memory_logits(sequences, self._language_model.vocab_size, pass_mode)
        _check_tokens_list(sequences, tokens_list)

        # Perform inference in model to compute the logits
        inputs = self._language_model.process_sequences_and_tokens(sequences)
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        # Remove padded logits
        lengths = [len(sequence) for sequence in sequences]
        logits = [logit[:length, :] for logit, length in zip(list(logits), lengths)]
        # Set to -inf logits that correspond to tokens that are not in tokens list
        vocabulary_mask = torch.from_numpy(self.get_vocabulary_mask(tokens_list))
        # Avoid printing warnings
        np.seterr(divide="ignore")

        masked_logits = []
        for logit in logits:
            masked_logit = logit + torch.from_numpy(
                np.tile(np.log(vocabulary_mask), (logit.shape[0], 1))
            )
            masked_logits.append(masked_logit)
        # Use softmax to compute probabilities frm logits
        # Due to the -inf, probs of tokens that are not in token list will be zero
        softmax = torch.nn.Softmax(dim=-1)
        probabilities = [softmax(logits) for logits in masked_logits]

        def _get_probabilities_dict(probs: torch.Tensor) -> Dict[str, float]:
            return {
                token: float(probs[i].cpu().numpy())
                for i, token in enumerate(self._language_model.model_vocabulary)
                if token in tokens_list
            }

        probabilities_dict = [
            {key: _get_probabilities_dict(value) for key, value in dict(enumerate(probs)).items()}
            for probs in probabilities
        ]
        return probabilities_dict

    def compute_loglikelihood(
        self,
        sequences: Union[List[str], str],
        batch_size: int = 1,
        tokens_list: List[str] = None,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> List[float]:
        """Function that computes loglikelihoods of sequences.
        It returns a list of float values.

        Both ProtBert and ESM models have more tokens than the 20 natural amino-acids
        (for instance MASK or PAD tokens). It might not be of interest to take these
        tokens into account when computing probabilities or log-likelihood. By default
        we remove them and compute probabilities only over the 20 natural amino-acids.
        This behavior can be overridden through the tokens_list argument that enable
        the user to choose the tokens to consider when computing probabilities.

        Args:
            sequences: List of sequences
            batch_size: Batch size
            tokens_list: List of tokens to consider
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            silent : display or not progress bar

        Returns:
            List[float]: list of log-likelihoods, one per sequence
        """
        probabilities = self.compute_probabilities(
            sequences, batch_size, tokens_list, pass_mode, silent
        )
        log_likelihoods = []
        for sequence, probabilities_dict in zip(sequences, probabilities):
            log_likelihood = np.sum(
                [np.log(probabilities_dict[i][sequence[i]]) for i in range(len(sequence))]
            )
            log_likelihoods.append(float(log_likelihood))
        return log_likelihoods

    def compute_embeddings(
        self,
        sequences: Union[List[str], str],
        batch_size: int = 1,
        pool_mode: Tuple[str, ...] = ("cls", "mean", "full"),
        silent: bool = False,
    ) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """Function that computes embeddings of sequences.

        The embedding of one sequence has a shape (sequence_length, embedding_size)
        where embedding_size equals 768 or 1024., thus we may want to use an aggregation
        function specified in pool_mode to aggregate the tensor on the num_tokens dimension.
        It might for instance avoid blowing the machine RAM when computing embeddings
        for a large number of sequences.

        'mean' signifies that we take the mean over the num_tokens dimension. 'cls'
        means that only the class token embedding is used.

        This function returns a dictionary of lists. The dictionary will have one key
        per pool-mode that has been specified. The corresponding value is a list of
        embeddings, one per sequence in sequences.

        Args:
            sequences: List of sequences or path of fasta file
            batch_size: Batch size
            pool_mode: Mode of pooling ('cls', 'mean', 'full')
            silent : whereas to display or not progress bar

        Returns:
             Dict[str, List[np.ndarray]]: dict matching pool-mode and list of embeddings
        """
        if isinstance(sequences, str):
            sequences = load_fasta(sequences)
        _check_sequence(sequences, self._model_dir, 1024)
        _check_memory_embeddings(sequences, self._language_model.embeddings_size, pool_mode)

        # Get the sequences lengths
        lengths = [len(sequence) for sequence in sequences]
        # Compute a forward pass to get the embeddings
        inputs = self._language_model.process_sequences_and_tokens(sequences)
        _, embeddings = self._model_evaluation(inputs, batch_size=batch_size, silent=silent)
        embeddings = [emb.cpu().numpy() for emb in embeddings]
        # Remove class token and padding
        filtered_embeddings = [
            emb[1 : (length + 1), :] for emb, length in zip(list(embeddings), lengths)
        ]
        # Keep class token only
        cls_embeddings = [emb[0, :] for emb in list(embeddings)]
        embeddings_dict = {}
        # Keep only what's necessary
        if "full" in pool_mode:
            embeddings_dict["full"] = filtered_embeddings
        if "cls" in pool_mode:
            embeddings_dict["cls"] = np.stack(cls_embeddings)
        if "mean" in pool_mode:
            embeddings_dict["mean"] = np.stack([np.mean(e, axis=0) for e in filtered_embeddings])
        return embeddings_dict

    def compute_accuracy(
        self,
        sequences: Union[List[str], str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> float:
        """Compute model accuracy from the input sequences

        Args:
            sequences (Union[List[str],str]): list of sequence or fasta file
            batch_size ([type], optional): [description]. Defaults to 1.
            pass_mode ([type], optional): [description]. Defaults to "forward".
            silent : whereas to display or not progress bar

        Returns:
            float: model's accuracy over the given sequences
        """
        if isinstance(sequences, str):
            sequences = load_fasta(sequences)
        _check_sequence(sequences, self._model_dir, 1024)
        _check_memory_logits(sequences, self._language_model.vocab_size, pass_mode)

        # Perform inference in model to compute the logits
        inputs = self._language_model.process_sequences_and_tokens(sequences)
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        # Get length of sequence
        labels = inputs["input_ids"]

        # Get the predicted labels
        predicted_labels = torch.argmax(logits, dim=-1)
        # Compute the accuracy
        accuracy = float(torch.mean(torch.eq(predicted_labels, labels).float()))
        return accuracy

    def load_model(self, model_dir: str, map_location=None):
        """Load state_dict a finetune pytorch model ro a checkpoint directory

        More informations about how to load a model with map_location:
            https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        Args:
            model_dir: path file of the pt model or checkpoint.
                       the checkpoint should be a pytorch model checkpoint
        """
        if not os.path.isfile(model_dir):
            raise FileNotFoundError

        if self._multi_gpus:
            ray.get(
                [worker._load_model.remote(model_dir, map_location) for worker in self._workers]
            )
            pass
        else:
            self._language_model._load_model(model_dir, map_location)  # type: ignore

    def _save_model(self, exp_path: str, lightning_model: pl.LightningModule):
        """Save pytorch model in pytorch-lightning logs directory
        Args:
            exp_path (str): path of the experiments directory in the logs
        """
        version = get_logs_version(exp_path)
        model_dir = self._model_dir.replace("/", "_")
        if version is not None:
            save_name = os.path.join(exp_path, version, model_dir + "_finetuned.pt")
        else:
            save_name = os.path.join(exp_path, model_dir + "_finetuned.pt")
        torch.save(lightning_model.model.state_dict(), save_name)
        log.info("Model save at %s." % save_name)

        return save_name

    def finetune(
        self,
        train_sequences: Union[List[str], str],
        lr: float = 1.0e-5,
        warmup_updates: int = 1024,
        warmup_init_lr: float = 1e-7,
        epochs: int = 10,
        batch_size: int = 2,
        acc_batch_size: int = 256,
        masking_ratio: float = 0.025,
        masking_prob: float = 0.8,
        random_token_prob: float = 0.15,
        toks_per_batch: int = 2048,
        filter_len: Optional[int] = None,
        accelerator: str = "ddp",
        amp_level: str = "O2",
        precision: int = 16,
        logs_save_dir: str = "logs",
        logs_name_exp: str = "finetune_masked",
        checkpoint: Optional[str] = None,
        save_last_checkpoint: bool = True,
    ):
        """Function to finetune a model on a specific dataset

        This function will finetune the choosen model on a dataset of
        sequences with pytorch ligthening. You can modify the masking ratio of AA
        in the arguments for better convergence.
        Be careful with the accelerator that you use. DDP accelerator will
        launch multiple python process and do not be use in a notebook.

        More informations on GPU/accelerator compatibility here :
            https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
        The wisest choice would be to use DDP for multi-gpu training.

        Args:
            train_sequences : Could be a list of sequences or the path of a
                              fasta file with multiple seqRecords
            lr : learning rate for training phase. Defaults to 1.0e-5.
            warmup_updates : Number of warming updates, number of step while increasing
            the leraning rate. Defaults to 1024.
            warmup_init_lr :  Initial lr for warming_update. Defaults to 1e-7.
            epochs :  number of epoch for training. Defaults to 10.
            batch_size :  mean number of sequence to consider in a batch. Defaults to 2.
            acc_batch_size : accumulated batch size Defaults to 2048.
            masking_ratio : ratio of tokens to be masked. Defaults to 0.025.
            masking_prob :  probability that the chose token is replaced with a mask token.
                            Defaults to 0.8.
            random_token_prob : probability that the chose token is replaced with a random token.
                                Defaults to 0.1.
            toks_per_batch: Maximum number of token to consider in a batch.Defaults to 2048.
                            This argument will set the number of sequences in a batch, which
                            is dynamically computed. Batch size use accumulate_grad_batches
                            to compute accumulate_grad_batches parameter.
            extra_toks_per_seq: Defaults to 2,
            filter_len : Size of sequence to filter. Defaults to None. (NOT USED)
            accelerator: type of accelerator for mutli-gpu processing (DPP recommanded)
            amp_level: allow mixed precision. Defaults to '02'
            precision: reducing precision allows to decrease the GPU memory needed.
                       Defaults to 16 (float16)
            logs_save_dir : Defaults directory to logs.
            logs_name_exp: Name of the experience in the logs.
            checkpoint : Path to a checkpoint file to restore training session.
            save_last_checkpoint: Save last checkpoint and 2 best trainings models
                                  to restore training session. Take a large amout of time
                                  and memory.
        """
        if isinstance(train_sequences, str):
            train_sequences = load_fasta(train_sequences)

        # Free resources used by ray before finetuning with Lightning
        del self._workers

        fit_model = self._language_model.model  # type: ignore
        alphabet = self._language_model.get_alphabet_dataloader()

        extra_toks_per_seq = int(alphabet.prepend_bos) + int(alphabet.append_eos)
        lightning_model = LightningModule(
            model=fit_model,
            alphabet=alphabet,
            lr=lr,
            warmup_updates=warmup_updates,
            warmup_init_lr=warmup_init_lr,
            warmup_end_lr=lr,
        )

        data_module = BioDataModule(
            train_sequences,
            alphabet,
            filter_len,
            masking_ratio,
            masking_prob,
            random_token_prob,
            toks_per_batch,
            extra_toks_per_seq,
        )

        if self._num_gpus == 0:
            raise ValueError("You try to train a transformers without GPU.")

        logger = CSVLogger(logs_save_dir, name=logs_name_exp)
        checkpoint_callback = None

        if save_last_checkpoint:
            checkpoint_callback = [
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=2,
                    mode="max",
                    monitor="val_acc",
                    every_n_val_epochs=3,
                )
            ]

        trainer = Trainer(
            gpus=self._num_gpus,
            amp_level=amp_level,
            precision=precision,
            accumulate_grad_batches=acc_batch_size // batch_size,
            max_epochs=epochs,
            logger=logger,
            accelerator=accelerator,
            replace_sampler_ddp=False,
            resume_from_checkpoint=checkpoint,
            callbacks=checkpoint_callback,
        )

        trainer.fit(lightning_model, data_module)

        save_path = str(Path(join(logs_save_dir, logs_name_exp)).resolve())
        if accelerator == "ddp":
            rank = os.environ.get("LOCAL_RANK", None)
            rank = int(rank) if rank is not None else None  # type: ignore
            if rank == 0:
                save_name = self._save_model(save_path, lightning_model)
        else:
            save_name = self._save_model(save_path, lightning_model)

        # Load new model
        self._language_model._load_model(save_name)

        if self._multi_gpus:
            # Create ray workers as they have been deleted at the beginning
            self._workers = [
                self._ray_cls.remote(model_dir=self._model_dir, device="cuda:0")
                for _ in range(self._num_gpus)
            ]
            # Load new model one ach worker
            ray.get([worker._load_model.remote(save_name) for worker in self._workers])

        log.info("Training completed.")

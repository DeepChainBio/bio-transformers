"""
This script defines a parent class for transformers, for which child classes which are
specific to a given transformers implementation can inherit.
It allows to derive probabilities, embeddings and log-likelihoods based on inputs
sequences, and displays some properties of the transformer model.
"""
import os
from abc import ABC, abstractmethod
from os.path import join
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.tensor
from biotransformers.lightning_utils.data import (
    AlphabetDataLoader,
    convert_ckpt_to_statedict,
)
from biotransformers.utils.constant import NATURAL_AAS_LIST
from biotransformers.utils.gpus_utils import set_device
from biotransformers.utils.logger import logger  # noqa
from biotransformers.utils.utils import (
    _check_memory_embeddings,
    _check_memory_logits,
    _check_sequence,
    get_logs_version,
    load_fasta,
)

# from esm.data import read_fasta
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.nn import DataParallel
from tqdm import tqdm

from ..lightning_utils.data import BioDataModule
from ..lightning_utils.models import LightningModule

log = logger("transformers_wrapper")


class TransformersWrapper(ABC):
    """
    Abstract class that uses pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(
        self,
        model_dir: str,
        _device: Optional[str] = None,
        multi_gpu: bool = False,
        mask_bool: bool = False,
    ):
        """Initialize Transformers wrapper

        Args:
            model_dir: name directory of the pretrained model
            _device: type of device to use (cpu or cuda).
            multi_gpu: turn on to True to use multigpu
            mask_bool: Wether to use mask or not for inference.
        """
        _device, _multi_gpu = set_device(_device, multi_gpu)

        self._device = torch.device(_device)
        self.multi_gpu = _multi_gpu
        self.model_dir = model_dir
        self.mask_bool = mask_bool

    @property
    def model_id(self) -> str:
        """Model ID, as specified in the model directory"""
        return self.model_dir.lower()

    @property
    @abstractmethod
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""

    @property
    @abstractmethod
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size"""

    @property
    @abstractmethod
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""

    @property
    @abstractmethod
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""

    @property
    @abstractmethod
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string).
        Returns an empty string if no such token"""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string).
        Returns an empty string if no such token."""

    @property
    @abstractmethod
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists"""

    @property
    @abstractmethod
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""

    @property
    @abstractmethod
    def embeddings_size(self) -> int:
        """Returns size of the embeddings"""

    def get_vocabulary_mask(self, tokens_list: List[str]) -> np.ndarray:
        """Returns a mask ove the model tokens."""
        # Compute a mask over vocabulary to decide which value to keep or not
        vocabulary_mask = np.array(
            [1.0 if token in tokens_list else 0.0 for token in self.model_vocabulary]
        )
        return vocabulary_mask

    @abstractmethod
    def _process_sequences_and_tokens(
        self, sequences_list: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model used"""

    @abstractmethod
    def _model_pass(
        self, model_inputs: Dict[str, torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Function which computes logits and embeddings based on a list of sequences"""

    @abstractmethod
    def _get_alphabet_dataloader(self) -> AlphabetDataLoader:
        """Function to build custom alphanbet"""

    def _get_num_batch_iter(self, model_inputs: Dict[str, Any], batch_size: int) -> int:
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
            for i in range(1, sum(binary_mask) - self.does_end_token_exist * 1):
                mask_sequence = torch.tensor(
                    sequence[:i].tolist()
                    + [self.token_to_id(self.mask_token)]
                    + sequence[i + 1 :].tolist(),
                    dtype=torch.int64,
                )
                new_input_ids.append(mask_sequence)
                new_attention_mask.append(binary_mask)
                new_token_type_ids.append(zeros)
                mask_id.append(i)
            mask_ids.append(mask_id)
        model_inputs["input_ids"] = torch.stack(new_input_ids)
        model_inputs["attention_mask"] = torch.stack(new_attention_mask)
        model_inputs["token_type_ids"] = torch.stack(new_token_type_ids)
        return model_inputs, mask_ids

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
        inf_tensor = -float("Inf") * torch.ones(
            [1, model_outputs.shape[2]], dtype=torch.float32
        )
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
        # Initialize logits and embeddings before looping over batches
        logits = torch.Tensor()  # [num_seqs, max_len_seqs+1, vocab_size]
        embeddings = torch.Tensor()  # [num_seqs, max_len_seqs+1, embedding_size]

        for batch_inputs in tqdm(
            self._generate_chunks(model_inputs, batch_size),
            total=self._get_num_batch_iter(model_inputs, batch_size),
            disable=silent,
        ):
            batch_logits, batch_embeddings = self._model_pass(batch_inputs)

            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)
            logits = torch.cat((logits, batch_logits), dim=0)

        return logits, embeddings

    def _compute_logits(
        self,
        model_inputs: Dict[str, torch.Tensor],
        batch_size: int,
        pass_mode: str,
        **kwargs
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
            logits, _ = self._model_evaluation(
                model_inputs, batch_size=batch_size, **kwargs
            )
            logits = self._gather_masked_outputs(logits, masked_ids_list)
        elif pass_mode == "forward":
            logits, _ = self._model_evaluation(
                model_inputs, batch_size=batch_size, **kwargs
            )
        return logits

    def compute_logits(
        self,
        sequences: Union[List[str], str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> List[np.ndarray]:
        """Function that computes the logits from sequences.

        It returns a list of logits for each sequence. Each sequence in the list
        contains only the amino acid of interest.

        Args:
            sequences_list: List of sequences
            batch_size: number of sequences to consider for the forward pass
            pass_mode: Mode of model evaluation ('forward' or 'masked')


        Returns:
            List[np.ndarray]: logits in np.ndarray format
        """

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        _check_sequence(sequences, self.model_dir, 1024)
        _check_memory_logits(sequences, self.vocab_size, pass_mode)

        # Perform inference in model to compute the logits
        inputs = self._process_sequences_and_tokens(sequences)
        labels = inputs["inputs_ids"]
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)

        # Remove padded logits
        lengths = [len(sequence) for sequence in sequences]
        logits = [logit[:length, :] for logit, length in zip(list(logits), lengths)]
        labels = [label[:length, :] for label, length in zip(list(labels), lengths)]

        # Keep only corresponding to amino acids that are in the sequence
        logits = [
            torch.gather(logit, dim=-1, index=label).numpy()
            for logit, label in zip(logits, labels)
        ]

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

        Args:
            sequences_list: List of sequences
            batch_size: number of sequences to consider for the forward pass
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            tokens_list: List of tokens to consider
            silent : display or not progress bar
        Returns:
            List[Dict[int, Dict[str, float]]]: dictionaries of probabilities per seq
        """
        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        tokens_list = NATURAL_AAS_LIST if tokens_list is None else tokens_list

        _check_sequence(sequences, self.model_dir, 1024)
        _check_memory_logits(sequences, self.vocab_size, pass_mode)

        # Perform inference in model to compute the logits
        inputs = self._process_sequences_and_tokens(sequences)
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)

        # Remove padded logits
        lengths = [len(sequence) for sequence in sequences]
        logits = [logit[:length, :] for logit, length in zip(list(logits), lengths)]

        # Set to -inf logits that correspond to tokens that are not in tokens list
        vocabulary_mask = self.get_vocabulary_mask(tokens_list)
        masked_logits = []
        for logit in logits:
            masked_logit = logit + np.tile(np.log(vocabulary_mask), (logit.shape[0], 1))
            masked_logits.append(masked_logit)

        # Use softmax to compute probabilities frm logits
        # Due to the -inf, probs of tokens that are not in token list will be zero
        softmax = torch.nn.Softmax(dim=-1)
        probabilities = [softmax(logits) for logits in masked_logits]

        def _get_probabilities_dict(probs: torch.Tensor) -> Dict[str, float]:
            return {
                token: float(probs[i].cpu().numpy())
                for i, token in enumerate(self.model_vocabulary)
                if token in tokens_list
            }

        probabilities_dict = [
            {
                key: _get_probabilities_dict(value)
                for key, value in dict(enumerate(probs)).items()
            }
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
        """Function that computes loglikelihoods of sequences

        Args:
            sequences: List of sequences
            batch_size: Batch size
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            tokens_list: List of tokens to consider

        Returns:
            List[float]: list of log-likelihoods, one per sequence
        """
        probabilities = self.compute_probabilities(
            sequences, batch_size, tokens_list, pass_mode, silent
        )
        log_likelihoods = []
        for sequence, probabilities_dict in zip(sequences, probabilities):
            log_likelihood = np.sum(
                [
                    np.log(probabilities_dict[i][sequence[i]])
                    for i in range(len(sequence))
                ]
            )
            log_likelihoods.append(float(log_likelihood))

        return log_likelihoods

    def compute_embeddings(
        self,
        sequences: Union[List[str], str],
        batch_size: int = 1,
        pool_mode: Tuple[str, ...] = ("cls", "mean", "full"),
        silent: bool = False,
    ) -> Dict[str, List[np.ndarray]]:
        """Function that computes embeddings of sequences.

        The embedding has a size (n_sequence, num_tokens, embeddings_size) so we use
        an aggregation function specified in pool_mode to aggregate the tensor on
        the num_tokens dimension. 'mean' signifies that we take the mean over the
        num_tokens dimension.

        Args:
            sequences: List of sequences or path of fasta file
            batch_size: Batch size
            pool_mode: Mode of pooling ('cls', 'mean', 'full')
            silent : whereas to display or not progress bar
        Returns:
             Dict[str, List[np.ndarray]]: todo: complete it
        """

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        _check_sequence(sequences, self.model_dir, 1024)
        _check_memory_embeddings(sequences, self.embeddings_size, pool_mode)

        # Get the sequences lengths
        lengths = [len(sequence) for sequence in sequences]

        # Compute a forward pass to get the embeddings
        inputs = self._process_sequences_and_tokens(sequences)
        _, embeddings = self._model_evaluation(
            inputs, batch_size=batch_size, silent=silent
        )

        # Remove class token and padding
        filtered_embeddings = [
            e[1 : (length + 1), :] for e, length in zip(list(embeddings), lengths)
        ]

        # Keep class token only
        cls_embeddings = [e[0, :] for e in list(embeddings)]

        embeddings_dict = {}
        # Keep only what's necessary
        if "full" in pool_mode:
            embeddings_dict["full"] = filtered_embeddings
        if "cls" in pool_mode:
            embeddings_dict["cls"] = cls_embeddings
        if "mean" in pool_mode:
            embeddings_dict["mean"] = [np.mean(e, axis=0) for e in filtered_embeddings]

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

        Returns:
            float: model's accuracy over the given sequences
        """
        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        _check_sequence(sequences, self.model_dir, 1024)
        _check_memory_logits(sequences, self.vocab_size, pass_mode)

        # Perform inference in model to compute the logits
        inputs = self._process_sequences_and_tokens(sequences)
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        labels = inputs["inputs_ids"]
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

        if model_dir.endswith(".pt"):
            load_model = torch.load(model_dir)
            log.info("Load model %s" % model_dir)
        elif model_dir.endswith(".ckpt"):
            load_model = convert_ckpt_to_statedict(torch.load(model_dir)["state_dict"])
            log.info("Load checkpoint %s" % model_dir)
        else:
            raise ValueError("Expecting a .pt or .ckpt file")

        if self.multi_gpu:
            self.model.module.load_state_dict(load_model, map_location)  # type: ignore
        else:
            self.model.load_state_dict(load_model, map_location)  # type: ignore
            self.model.eval()  # type: ignore

    def save_model(self, exp_path: str, lightning_model: pl.LightningModule):
        """Save pytorch model in logs directory

        Args:
            exp_path (str): path of the experiments directory in the logs
        """
        version = get_logs_version(exp_path)
        model_dir = self.model_dir.replace("/", "_")
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
        filter_len=1024,
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
            batch_size :  number of sequence to consider in a batch. Defaults to 2.
            acc_batch_size : accumulated batch size Defaults to 2048.
            masking_ratio : ratio of tokens to be masked. Defaults to 0.025.
            masking_prob :  probability that the chose token is replaced with a mask token.
                            Defaults to 0.8.
            random_token_prob : probability that the chose token is replaced with a random token.
                                Defaults to 0.1.
            toks_per_batch: Maximum number of token to consider in a batch.Defaults to 2048.
                            This argument will set the number of sequences in a batch, which
                            is dynamically computed. Batch size use accumulate_grad_batches to compute
                            accumulate_grad_batches parameter.
            extra_toks_per_seq: Defaults to 2,
            filter_len : Size of sequence to filter. Defaults to 1024. (NOT USED)
            accelerator: type of accelerator for mutli-gpu processing (DPP recommanded)
            amp_level: allow mixed precision. Defaults to '02'
            precision: reducing precision allows to decrease the GPU memory needed.
                       Defaults to 16 (float16)
            logs_save_dir : Defaults directory to logs.
            logs_name_exp: Name of the experience in the logs.
            checkpoint : Path to a checkpoint file to restore training session.
            save_last_checkpoint: Save last checkpoint and 2 best trainings models
                                  to restore training session. Take a large amout of time and memory.
        """
        if isinstance(train_sequences, str):
            train_sequences = load_fasta(train_sequences)
        _check_sequence(train_sequences, self.model_dir, 1024)  # noqa: ignore

        fit_model = self.model.module if self.multi_gpu else self.model  # type: ignore
        alphabet = self._get_alphabet_dataloader()

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
            batch_size,
            masking_ratio,
            masking_prob,
            random_token_prob,
            toks_per_batch,
            extra_toks_per_seq,
        )

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
        else:
            log.warning("You try to train a transformers without GPU.")
            return

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
            gpus=n_gpus,
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
                self.save_model(save_path, lightning_model)
        else:
            self.save_model(save_path, lightning_model)

        if self.multi_gpu:
            self.model = DataParallel(lightning_model.model).to(self._device)
        else:
            self.model = lightning_model.model.to(self._device)

        log.info("Training completed.")

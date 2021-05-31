"""
This script defines a class which inherits from the TransformersWrapper class, and is
specific to the ESM model developed by FAIR (https://github.com/facebookresearch/esm).
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import esm
import torch
from biotransformers.utils.constant import DEFAULT_ESM_MODEL, ESM_LIST
from biotransformers.utils.logger import logger  # noqa
from biotransformers.utils.utils import _check_sequence, load_fasta
from biotransformers.wrappers.transformers_wrappers import TransformersWrapper

# from esm.data import read_fasta
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.nn import DataParallel

from ..lightning_utils.data import BioDataModule
from ..lightning_utils.models import LightningESM

log = logger("esm_wrapper")


class ESMWrapper(TransformersWrapper):
    """
    Class that uses an ESM type of pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(self, model_dir: str, device, multi_gpu):

        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. "
                f"Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        super().__init__(model_dir, _device=device, multi_gpu=multi_gpu)

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_dir)
        self.num_layers = self.model.num_layers
        self.hidden_size = self.model.args.embed_dim

        if self.multi_gpu:
            self.model = DataParallel(self.model).to(self._device)
        else:
            self.model = self.model.to(self._device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @property
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""
        return self.model_id

    @property
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""
        return list(self.alphabet.tok_to_idx.keys())

    @property
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size"""
        return len(list(self.alphabet.tok_to_idx.keys()))

    @property
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.mask_idx]  # "<mask>"

    @property
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.padding_idx]  # "<pad>"

    @property
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.cls_idx]  # "<cls>"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.eos_idx]  # "<eos>"

    @property
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists"""
        return self.alphabet.append_eos

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""
        return lambda x: self.alphabet.tok_to_idx[x]

    @property
    def embeddings_size(self):
        """Returns size of the embeddings"""
        return self.hidden_size

    def _process_sequences_and_tokens(
        self, sequences_list: List[str], tokens_list: List[str]
    ) -> Tuple[Dict[str, torch.tensor], torch.tensor, List[int]]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        tokens = []
        for token in tokens_list:
            if token not in self.model_vocabulary:
                print("Warnings; token", token, "does not belong to model vocabulary")
            else:
                tokens.append(self.token_to_id(token))

        _, _, all_tokens = self.batch_converter(
            [("", sequence) for sequence in sequences_list]
        )

        all_tokens = all_tokens.to("cpu")

        encoded_inputs = {
            "input_ids": all_tokens,
            "attention_mask": 1 * (all_tokens != self.token_to_id(self.pad_token)),
            "token_type_ids": torch.zeros(all_tokens.shape),
        }
        return encoded_inputs, all_tokens, tokens

    def _model_pass(
        self, model_inputs: Dict[str, torch.tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        Args:
            model_inputs (Dict[str, torch.tensor]): [description]

        Returns:
            Tuple[torch.tensor, torch.tensor]:
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        last_layer = self.num_layers - 1
        with torch.no_grad():
            model_outputs = self.model(
                model_inputs["input_ids"].to(self._device), repr_layers=[last_layer]
            )
            logits = model_outputs["logits"].detach().cpu()
            embeddings = model_outputs["representations"][last_layer].detach().cpu()

        return logits, embeddings

    def train_masked(
        self,
        train_sequences: Union[List[str], str],
        lr: float = 1.0e-5,
        warmup_updates: int = 10,
        warmup_init_lr: float = 1e-7,
        epochs: int = 10,
        batch_size: int = 2,
        acc_batch_size: int = 2048,
        masking_ratio: float = 0.025,
        masking_prob: float = 0.8,
        random_token_prob: float = 0.15,
        toks_per_batch: int = 128,
        filter_len=1024,
        accelerator: str = "ddp",
        amp_level: str = "O2",
        precision: int = 16,
        logs_save_dir: str = "logs",
        logs_name_exp: str = "finetune_masked",
        checkpoint: Optional[str] = None,
        save_last_checkpoint=True,
    ):
        """Function to finetuned a model on a specific dataset

        This function will finetune a the choosen model on a dataset of
        sequences with pytorch ligthening. You can modify the masking ratio of AA
        in the arguments.
        Be careful with the accelerator that you use. Could meet some troubles depending
        on the type of GPU you use.

        More informations on GPU/accelerator compatibility here :
            https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
        The wisest choice would be to use DDP for multi-gpu training.

        Args:
            train_sequences : Could be a list of sequences or the path of a
                              fasta file with multiple seqRecords
            lr : learning rate for training phase. Defaults to 1.0e-5.
            warmup_updates : Number of warming updates. Defaults to 10.
            warmup_init_lr :  Initial lr for warming_update. Defaults to 1e-7.
            epochs :  number of epoch for training. Defaults to 10.
            batch_size :  number of sequence to consider in a batch. Defaults to 2.
            acc_batch_size : accumulated batch size Defaults to 2048.
            masking_ratio : ratio of tokens to be masked. Defaults to 0.025.
            masking_prob :  probability that the chose token is replaced with a mask token.
                            Defaults to 0.8.
            random_token_prob : probability that the chose token is replaced with a random token.
                                Defaults to 0.1.
            toks_per_batch: Maximum number of token to consider in a batch.Defaults to 128,
            extra_toks_per_seq: Defaults to 2,
            filter_len : Size of sequence to filter. Defaults to 1024.
            accelerator: type of accelerator for mutli-gpu processing
            amp_level: allow mixed precision. Defaults to '02'
            precision: reducing precision allows to decrease the GPU memory needed.
                       Defaults to 16 (float16)
            logs_save_dir : Defaults to logs.
            logs_name_exp: Name of the experience in the logs
            checkpoint : Path to a checkpoint file to restore training session
            save_last_checkpoint: Save last checkpoint and 2 best trainings models
                                  to restore training session
        """
        if isinstance(train_sequences, str):
            train_sequences = load_fasta(train_sequences)
        _check_sequence(train_sequences, self.model_dir, 1024)  # noqa: ignore

        fit_model = self.model.module if self.multi_gpu else self.model

        extra_toks_per_seq = int(self.alphabet.prepend_bos) + int(
            self.alphabet.append_eos
        )
        lightning_model = LightningESM(
            model=fit_model,
            alphabet=self.alphabet,
            lr=lr,
            warmup_updates=warmup_updates,
            warmup_init_lr=warmup_init_lr,
            warmup_end_lr=lr,
        )

        data_module = BioDataModule(
            train_sequences,
            self.alphabet,
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
        else:
            checkpoint_callback = None

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

        exp_path = os.path.join(logs_save_dir, logs_name_exp)
        save_name = self.save_model(exp_path, lightning_model)
        log.info("Model save at %s." % save_name)

        if self.multi_gpu:
            self.model = DataParallel(lightning_model.model).to(self._device)
        else:
            self.model = lightning_model.model.to(self._device)

        log.info("Training completed.")

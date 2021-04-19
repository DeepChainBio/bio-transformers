"""
This script defines a class which inherits from the TransformersWrapper class, and is
specific to the ESM model developed by FAIR (https://github.com/facebookresearch/esm).
"""
from typing import Dict, List

import esm
import numpy as np
import torch
from tqdm import tqdm

from .transformers_wrappers import (
    NATURAL_AAS,
    TransformersInferenceConfig,
    TransformersModelProperties,
    TransformersWrapper,
)

# List all ESM models
esm_list = [
    "esm1_t34_670M_UR50S",
    "esm1_t34_670M_UR50D",
    "esm1_t34_670M_UR100",
    "esm1_t12_85M_UR50S",
    "esm1_t6_43M_UR50S",
    "esm1b_t33_650M_UR50S",
    "esm_msa1_t12_100M_UR50S",
]

# Define a default ESM model
DEFAULT_MODEL = "esm1b_t33_650M_UR50S"


class ESMWrapper(TransformersWrapper):
    """
    Class that uses an ESM type of pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(self, model_dir: str, device: str = None):

        if model_dir not in esm_list:
            print(
                f"Model dir '{model_dir}' not recognized. "
                f"Using '{DEFAULT_MODEL}' as default"
            )
            model_dir = DEFAULT_MODEL

        super().__init__(model_dir, _device=device)

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_dir)
        # TODO: use nn.Parallel to make parallel inference
        self.model = self.model.to(self._device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @property
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""
        return self.model_id

    @property
    def model_property(self) -> TransformersModelProperties:
        """Returns a class with model properties"""
        return TransformersModelProperties(
            num_sep_tokens=1, begin_token=True, end_token=False
        )

    @property
    def model_vocab_tokens(self) -> List[str]:
        """List of all vocabulary tokens to consider (as strings), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""
        voc = (
            self.vocab_token_list
            if self.vocab_token_list is not None
            else self.alphabet.all_toks
        )
        return voc

    @property
    def model_vocab_ids(self) -> List[int]:
        """List of all vocabulary IDs to consider (as ints), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""
        return [self.token_to_id(tok) for tok in self.model_vocab_tokens]

    @property
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.mask_idx]  # "<mask>"

    @property
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        return "<cls>"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string). This token doesn't
        exist in the case of ESM, thus we return an empty string."""
        return ""

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""
        return lambda x: self.alphabet.tok_to_idx[x]

    def compute_masked_input(
        self,
        sequence: str,
        seq_id: int,
        inference_config: TransformersInferenceConfig,
        max_seq_len: int = None,
    ) -> torch.tensor:
        """
        Function to compute a masked input based on a sequence, a sequence id,
        a configuration and the maximum sequence length of the batch.

        Args:
            sequence (str): [description]
            seq_id (int): [description]
            inference_config (TransformersInferenceConfig): [description]
            max_seq_len (int, optional): [description]. Defaults to None.

        Returns:
            torch.tensor: [description]
        """
        data = [("", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.squeeze()

        # Add padding based on the maximum sequence length in the batch
        if max_seq_len is not None:
            padding_length = max_seq_len - len(sequence)
            if padding_length > 0:
                padding_tensor = torch.full(
                    size=(padding_length,), fill_value=self.alphabet.padding_idx
                )
                batch_tokens = torch.cat((batch_tokens, padding_tensor))

        # Initialise all tokenized data with empty tensor
        data_all = torch.Tensor().long()
        if inference_config.mutation_dicts_list is None:
            # create masked sequences for all amino-acids
            # Start after the "beginning" token
            for i in range(1, len(batch_tokens)):
                masked_data = batch_tokens.clone()
                masked_data[i] = self.alphabet.tok_to_idx[self.mask_token]
                data_all = torch.cat((data_all, masked_data))
        else:
            for i in range(1, len(batch_tokens)):
                # Only include masks for mutated amino acids
                if (i - 1) in inference_config.mutation_dicts_list[seq_id].keys():
                    masked_data = batch_tokens.clone()
                    masked_data[i] = self.alphabet.tok_to_idx[self.mask_token]
                    data_all = torch.cat((data_all, masked_data))
        return data_all.unsqueeze(dim=0)

    def _compute_masked_output(
        self,
        sequences_list: list,
        batch_size: int,
        inference_config: TransformersInferenceConfig,
    ) -> Dict[torch.tensor, torch.tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by masking sequentially each amino-acid of the sequence (or only mutated amino-
        acids if specified this way), following a "masked inference" approach.

        Args:
            sequences_list (list): [description]
            batch_size (int): [description]
            inference_config (TransformersInferenceConfig): [description]

        Returns:
            Dict[torch.tensor, torch.tensor]:
                        * logits [num_seqs, max_len_seqs+1, vocab_size]
                        * embeddings : [num_seqs, max_len_seqs*(max_len_seqs+1), embedding_size]
        """

        # Initialize probabilities and embeddings before looping over batches
        logits = torch.Tensor()  # [num_seqs, max_len_seqs+1, vocab_size]
        embeddings = torch.Tensor()  # [num_seqs, max_len_seqs*(max_len_seqs+1),
        #  embedding_size]
        max_seq_len = max([len(seq) for seq in sequences_list])

        for seq_id, sequence in tqdm(enumerate(sequences_list)):

            data_all = self.compute_masked_input(
                sequence, seq_id, inference_config, max_seq_len
            )

            # Define batch size and number of iterations
            batch_size = len(data_all) if len(data_all) < batch_size else batch_size
            num_batch_iter = int(np.ceil(len(data_all) / batch_size))
            if seq_id == 0:
                print("Batch size: ", batch_size)
                print("Number of batch iterations: ", num_batch_iter)

            for batch_iter in range(num_batch_iter):
                data_batch = data_all[
                    batch_iter * batch_size : (batch_iter + 1) * batch_size, :
                ]
                data_batch = data_batch.to(self._device)
                # Extract per-residue representations of the last layer
                last_layer = self.model.num_layers - 1
                with torch.no_grad():
                    results = self.model(data_batch, repr_layers=[last_layer])

                # Extract embeddings of each position (or mutation) of each sequence
                new_embeddings = results["representations"][last_layer]
                new_embeddings = new_embeddings.detach().cpu()
                embeddings = torch.cat((embeddings, new_embeddings), dim=0)

                # Extract logits and remove first token for beginning of sentence
                new_logits = results["logits"].detach().cpu()
                logits = torch.cat((logits, new_logits), dim=0)

            # Update Out-Of-Vocabulary logits
            logits = self.update_oov_logits(logits)

        return {"logits": logits, "embeddings": embeddings}

    def _compute_forward_output(
        self,
        sequences_list: list,
        batch_size: int,
    ) -> Dict[torch.tensor, torch.tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        Args:
            sequences_list (list): [description]
            batch_size (int): [description]

        Returns:
            Dict[torch.tensor, torch.tensor]:
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        # Define number of iterations
        num_batch_iter = int(np.ceil(len(sequences_list) / batch_size))
        # Initialize probabilities and embeddings before looping over batches
        logits = torch.Tensor()  # [num_seqs, max_len_seqs+1, vocab_size]
        embeddings = torch.Tensor()  # [num_seqs, max_len_seqs+1, embedding_size]

        _, _, all_tokens = self.batch_converter(
            [("", sequence) for sequence in sequences_list]
        )

        for batch_tokens in tqdm(
            self._generate_chunks(all_tokens, batch_size), total=num_batch_iter
        ):
            batch_tokens = batch_tokens.to(self._device)
            last_layer = self.model.num_layers - 1

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[last_layer])

            # Also include first token embedding (for the beginning of the sentence)
            new_embeddings = results["representations"][last_layer]
            new_embeddings = new_embeddings.detach().cpu()
            embeddings = torch.cat((embeddings, new_embeddings), dim=0)

            #  Get the logits : token 0 is always a beginning-of-sequence token
            #  , so the first residue is token 1.
            new_logits = results["logits"].detach().cpu()
            logits = torch.cat((logits, new_logits), dim=0)

        # Update Out-Of-Vocabulary logits
        logits = self.update_oov_logits(logits)

        return {"logits": logits, "embeddings": embeddings}

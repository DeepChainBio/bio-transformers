"""
This script defines a class which inherits from the TransformersWrapper class, and is
specific to the Rostlab models (eg ProtBert and ProtBert-BFD) developed by
hugging face
- ProtBert: https://huggingface.co/Rostlab/prot_bert
- ProtBert BFD: https://huggingface.co/Rostlab/prot_bert_bfd
"""
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer, pipeline

from .transformers_wrappers import (
    NATURAL_AAS,
    TransformersInferenceConfig,
    TransformersModelProperties,
    TransformersWrapper,
)

rostlab_list = ["Rostlab/prot_bert", "Rostlab/prot_bert_bfd"]
DEFAULT_MODEL = "Rostlab/prot_bert"


class RostlabWrapper(TransformersWrapper):
    """
    Class that uses a rostlab type of pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(self, model_dir: str, device: str = None):

        if model_dir not in rostlab_list:
            print(
                f"Model dir '{model_dir}' not recognized. "
                f"Using '{DEFAULT_MODEL}' as default"
            )
            model_dir = DEFAULT_MODEL

        super().__init__(model_dir, _device=device)

        self.tokenizer = BertTokenizer.from_pretrained(
            model_dir, do_lower_case=False, padding=True
        )
        self.model_dir = model_dir
        self.model = None
        self.mask_pipeline = None

    @property
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""
        return self.model_id.replace("rostlab/", "")

    @property
    def model_property(self) -> TransformersModelProperties:
        """Returns a class with model properties"""
        return TransformersModelProperties(
            num_sep_tokens=2, begin_token=True, end_token=True
        )

    @property
    def model_vocab_tokens(self) -> List[str]:
        """List of all vocabulary tokens to consider (as strings), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""
        voc = (
            self.vocab_token_list
            if self.vocab_token_list is not None
            else self.tokenizer.vocab
        )
        return voc

    @property
    def model_vocab_ids(self) -> List[int]:
        """List of all vocabulary IDs to consider (as ints), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""
        vocab_ids = [
            self.tokenizer.convert_tokens_to_ids(tok) for tok in self.model_vocab_tokens
        ]
        return vocab_ids

    @property
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        return self.tokenizer.mask_token  # "[MASK]"

    @property
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        return "[CLS]"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)."""
        return "[SEP]"

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""
        return lambda x: self.tokenizer.convert_tokens_to_ids(x)

    def inference_config(self) -> TransformersInferenceConfig:
        return TransformersInferenceConfig(False)

    def load_model(self):
        """Load model as attribute"""
        if self.model is None:
            self.model = BertForMaskedLM.from_pretrained(self.model_dir)
            self.model = self.model.eval().to(self._device)

    def load_pipeline(self):
        """Load pipeline as attribute"""
        if self.mask_pipeline is None:
            self.load_model()
            self.mask_pipeline = pipeline(
                "fill-mask",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=self.tokenizer.vocab_size,
                device=0,
            )

    def compute_masked_input(
        self,
        sequence: str,
        seq_id: int,
    ) -> List:
        """
        Function to compute a masked input based on a sequence, a sequence id,
        and a configuration.

        Args:
            sequence (str): [description]
            seq_id (int): [description]
            inference_config (TransformersInferenceConfig, optional):
                                                [description]. Defaults to None.

        Returns:
            List: [description]
        """
        inference_config = self.inference_config()

        # Compute the masked sequences to input the model
        if inference_config.mutation_dicts_list is None:
            # Compute masks for all tokens in the sequence
            masks = [sequence for _ in range(len(sequence))]
            for i in range(len(masks)):
                s = list(masks[i])
                s[i] = self.mask_token
                masks[i] = " ".join(s)
        else:
            # Only compute masks for mutated tokens
            masks = [
                sequence
                for _ in range(len(inference_config.mutation_dicts_list[seq_id]))
            ]
            for i, mut_aa_pos in enumerate(
                inference_config.mutation_dicts_list[seq_id].keys()
            ):
                s = list(masks[i])
                s[mut_aa_pos] = self.mask_token
                masks[i] = " ".join(s)
        return masks

    def _compute_masked_output(
        self,
        sequences_list: List[str],
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
                    * Logits: [num_seqs, max_len_seqs+2, vocab_size]
                    * Embeddings: None
        """

        # Load mask pipeline
        self.load_pipeline()

        # Initialize probs lists
        probs = []  # [num_seqs, max_len_seqs+2, vocab_size]
        # Mask pipeline doesn't return hidden states
        embeddings = None
        # maximum sequene length (for padding)
        max_seq_len = max([len(seq) for seq in sequences_list])
        # Loop over sequences
        for seq_id, sequence in tqdm(enumerate(sequences_list)):

            masks = self.compute_masked_input(
                sequence=sequence, seq_id=seq_id, inference_config=inference_config
            )

            # Define batch size and number of iterations for each sequence
            batch_size = len(masks) if len(masks) < batch_size else batch_size
            num_batch_iter = int(np.ceil(len(masks) / batch_size))
            if seq_id == 0:
                print("Batch size: ", batch_size)
                print("Number of batch iterations: ", num_batch_iter)

            # Loop over batches
            model_outs = []
            for batch_iter in range(num_batch_iter):
                masks_sublist = masks[
                    batch_iter * batch_size : (batch_iter + 1) * batch_size
                ]
                model_out = self.mask_pipeline(masks_sublist)
                # Avoids issue if only one sequence is sent to the model
                model_outs += [model_out] if len(masks_sublist) == 1 else model_out

            probs_seq = []
            for model_out in model_outs:
                token_ids = [model_out_aa["token"] for model_out_aa in model_out]
                token_scores = [model_out_aa["score"] for model_out_aa in model_out]
                aa_prob = np.array(token_scores)[np.argsort(token_ids)]
                probs_seq.append(aa_prob)
            # add padding if needed
            padding = (
                [aa_prob * 0] * (max_seq_len - len(sequence))
                if max_seq_len - len(sequence) > 0
                else []
            )
            probs_seq += padding
            probs.append(probs_seq)
            logits = torch.logit(torch.tensor(probs))
            # Update OOV logits
            logits = self.update_oov_logits(logits)
        return {"logits": logits, "embeddings": embeddings}

    def _compute_forward_output(
        self,
        sequences_list: List[str],
        batch_size: int,
    ) -> Dict[torch.tensor, torch.tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        Args:
            sequences_list (list): [description]
            batch_size (int): [description]
            inference_config (TransformersInferenceConfig): [description]

        Returns:
            Dict[torch.tensor, torch.tensor]:
                - Logits: [num_seqs, max_len_seqs+2, vocab_size]
                - Embeddings: [num_seqs, max_len_seqs+2, embedding_size]
        """

        # Load model and transfer to device
        self.load_model()

        # If local mutations only, for each sequence replace all mutated amino-acids
        # by a mask (there may be several masks in the same sentence)
        inference_config = self.inference_config()

        if (inference_config.mutation_dicts_list is not None) & (
            inference_config.all_masks_forward_local_bool is True
        ):
            input_list = list(sequences_list)
            for seq_id in range(len(sequences_list)):
                if len(inference_config.mutation_dicts_list[seq_id]) > 0:
                    mutated_tokens = inference_config.mutation_dicts_list[seq_id].keys()
                    masked_seq = list(input_list[seq_id])
                    for i, aa in enumerate(masked_seq):
                        masked_seq[i] = (
                            aa if i not in mutated_tokens else self.mask_token
                        )
                    input_list[seq_id] = masked_seq
        else:
            input_list = list(sequences_list)

        num_seqs = len(sequences_list)
        if inference_config.mutation_dicts_list is not None:
            print(
                "Mask all mutated tokens at once"
                if inference_config.all_masks_forward_local_bool
                else "All mutated tokens visible"
            )

        batch_size = num_seqs if num_seqs < batch_size else batch_size
        num_batch_iter = int(np.ceil(num_seqs / batch_size))

        logits = torch.Tensor()  # [num_seqs, max_len_seqs+2, vocab_size]
        embeddings = torch.Tensor()  # [num_seqs, max_len_seqs+2, embedding_size]

        input_sep_sequences_list = [" ".join(seq) for seq in input_list]
        encoded_inputs = self.tokenizer(
            input_sep_sequences_list, return_tensors="pt", padding=True
        )

        for batch_iter in tqdm(range(num_batch_iter)):
            batch_encoded_inputs = {
                key: value[batch_iter * batch_size : (batch_iter + 1) * batch_size].to(
                    self._device
                )
                for key, value in encoded_inputs.items()
            }
            output = self.model(**batch_encoded_inputs, output_hidden_states=True)
            new_logits = output.logits.detach().cpu()
            logits = torch.cat((logits, new_logits), dim=0)
            # Only keep track of the hidden states of the last layer
            new_embeddings = output.hidden_states[-1]
            new_embeddings = new_embeddings.detach().cpu()
            embeddings = torch.cat((embeddings, new_embeddings), dim=0)

        # Update OOV logits
        logits = self.update_oov_logits(logits)

        return {"logits": logits, "embeddings": embeddings}

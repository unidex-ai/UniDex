from typing import List

import torch

IMAGENET_STANDARD_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = torch.tensor([0.5, 0.5, 0.5])


def add_pcd_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    pcd_seq_len,
    pcd_token,
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{pcd_token * pcd_seq_len}{bos_token}{prefix_prompt}\n"

class VLAProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        tokenizer,
        num_pcd_tokens: int,
        max_seq_len: int,
        tokenizer_padding: str = "max_length",  #  # instead of truncating to longest
    ):
        super().__init__()

        self.pcd_seq_length = num_pcd_tokens
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.pcd_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        pcd: torch.FloatTensor,
        truncation: bool = True,
    ) -> dict:
        assert len(pcd) == len(
            text
        ), f"Received {len(pcd)} pointcloud for {len(text)} prompts."

        # Prepend a `self.pcd_seq_length` number of pcd tokens to the prompt
        input_strings = [
            add_pcd_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                pcd_seq_len=self.pcd_seq_length,
                pcd_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        output = {"pointcloud": pcd, **inputs}
        return output

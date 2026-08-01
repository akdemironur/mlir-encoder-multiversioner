#!/usr/bin/env python3
"""Validate E5 token fixtures without padding or rewriting them."""

from __future__ import annotations

import numpy as np


VOCAB_SIZE = 30522
TYPE_VOCAB_SIZE = 2


def validate_inputs(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    token_type_ids: np.ndarray,
) -> None:
    values = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    for name, value in values.items():
        if value.dtype != np.int64:
            raise ValueError(f"{name} must have dtype int64")
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1,S]")
    widths = {value.shape[1] for value in values.values()}
    if len(widths) != 1:
        raise ValueError("all three token inputs must have equal widths")
    if not np.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must contain only 0/1")
    if not np.any(attention_mask == 1):
        raise ValueError("attention_mask must contain at least one 1")
    if not np.all((input_ids >= 0) & (input_ids < VOCAB_SIZE)):
        raise ValueError("input_ids are outside the model vocabulary")
    if not np.all((token_type_ids >= 0) & (token_type_ids < TYPE_VOCAB_SIZE)):
        raise ValueError("token_type_ids are outside the model range")

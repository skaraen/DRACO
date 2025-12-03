#!/usr/bin/env python3

from torch import nn
from .core_original import MultiHeadedAttention as MultiHeadedAttentionOriginal
from .core import PositionwiseFeedForward, PositionalEncoding
from .modules import Encoder, EncoderLayer


def make_transformer(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    from copy import deepcopy
    c = deepcopy
    # attn = MultiHeadedAttention(h, d_model)
    attn = MultiHeadedAttentionOriginal(h, d_model)  # This is slightly faster (~4%)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # (self) attention block followed by a (positionwise) feed forward block
    enc_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)

    model = Encoder(enc_layer, N)

    # Seems like this was important
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

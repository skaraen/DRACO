#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from einops import einsum, rearrange
import torch.nn as nn
import torch.nn.functional as F
import math

from .helper import clones


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer: EncoderLayer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        # print('=== Encoder ===')
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.

    Hmmm, this is already in the form of pre-norm sublayer connection that is used in GPT-2!
    NOTE: In GPT-2, dropout is not applied except in the MLP block.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        self_attn_layer = lambda x: self.self_attn(x, x, x, mask)
        x = self.sublayer[0](x, self_attn_layer)
        return self.sublayer[1](x, self.feed_forward)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

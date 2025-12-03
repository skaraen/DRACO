#!/usr/bin/env python3
# From annotated Transformer (https://nlp.seas.harvard.edu/2018/04/03/attention.html)

import numpy as np
from einops import einsum, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .helper import clones


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'.

    Args:
        - query: (batch_size, num_heads, seq_len, d_k)
        - key:   (batch_size, num_heads, seq_len, d_k)
        - value: (batch_size, num_heads, seq_len, d_v)

    Returns:
        - out_vals: (batch_size, num_heads, seq_len, d_v)
        - p_attn: (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Attention logits over tokens (j) for the token i
    # (batch_size, num_heads, seq_len, d_k) => (batch_size, num_heads, seq_len, seq_len)
    scores = einsum(query, key, 'b h i d, b h j d -> b h i j') / math.sqrt(d_k)
    # scores = torch.matmul(query, key.transpose(-2, -1)) \
    #          / math.sqrt(d_k)
    if mask is not None:
        # print('mask', mask.shape, 'scores', scores.shape)
        # NOTE: Set -1e9 to the positions with mask == 0
        # After the softmax, attentions of these positions will be very close to zero
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax over j
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Value of the token i weighted by attention
    out_vals = einsum(p_attn, value, 'b h i j, b h j d -> b h i d')
    return out_vals, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # num heads
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        TODO: Fill this in
        Args:
            - query: (batch_size, seq_len, d_k)
            - key: (batch_size, seq_len, d_k)
            - value: (batch_size, seq_len, d_k)
            - mask: (batch_size, seq_len, seq_len)
        Returns:
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = rearrange(mask, 'b i j -> b () i j')
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => (h, d_k)
        # (batch_size, seq_len, d_model) => (batch_size, seq_len, d_k, num_heads)
        query, key, value = \
            [rearrange(l(x), 'b i (h d_k) -> b h i d_k', h=self.h, d_k=self.d_k)
             for l, x in zip(self.linears, (query, key, value))]


        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" and apply a final linear.
        x = rearrange(x, 'b h i d_k -> b i (h d_k)').contiguous()

        # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous() \
        #      .view(nbatches, -1, self.h * self.d_k)
        return self.final_linear(x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation.

    Args:
        - d_model: d_k x num_heads
        - d_ff: hidden layer size
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)],
                             requires_grad=False)
        return self.dropout(x)

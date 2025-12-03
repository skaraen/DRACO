#!/usr/bin/env python3

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch.nn as nn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

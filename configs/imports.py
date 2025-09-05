import math
import os
import traceback  # For detailed exception traces
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack

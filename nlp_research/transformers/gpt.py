'''Implementation of GPT model.'''
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import SequenceSummary
from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

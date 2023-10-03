'''Module for BERT model'''
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup


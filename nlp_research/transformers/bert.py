'''Module for BERT model'''
import torch
import torch.nn as nn
from transformers import (AdamW, BertConfig, BertForSequenceClassification,
                          BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup)


class Bert(nn.Module):
    '''BERT model'''
    def __init__(self, num_classes, freeze_bert=False):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        '''Forward pass'''
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        output = self.dropout(pooled_output)
        return self.classifier(output)

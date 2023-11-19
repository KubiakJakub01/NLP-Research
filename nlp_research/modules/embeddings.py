'''Module with different embeddings'''
import einops
import torch
from torch import nn


class Embeddings(nn.Module):
    '''Embeddings module'''

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pad_token_id: int,
        max_position_embeddings: int,
    ):
        '''Init Embeddings module
        
        Args:
            vocab_size (int): vocab size
            embedding_dim (int): embedding dim
            pad_token_id (int): pad token id
            max_position_embeddings (int): max position embeddings'''
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, embedding_dim
        )
        self.LayerNorm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''Forward pass for Embeddings module

        Args:
            input_ids (torch.Tensor): input ids

        Returns:
            torch.Tensor: embeddings'''
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

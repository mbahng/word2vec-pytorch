import torch.nn as nn 
from torch.nn import Module

class CBOW(Module): 

  def __init__(self, embedding_dim: int, vocab_size: int): 
    self.embedding_dim = embedding_dim 
    self.vocab_size = vocab_size  

    self.embedding = nn.Embedding(
      num_embeddings = self.vocab_size, 
      embedding_dim = self.embedding_dim, 
      max_norm = 1
    )

    self.linear = nn.Linear(
      in_features = self.embedding_dim, 
      out_features = self.vocab_size
    )

  def forward(self, x): 
    """Takes in a one-hot encoded vector in vocab""" 
    x = self.embedding(x) 
    x = x.mean(axis=1)
    x = self.linear(x) 
    return x


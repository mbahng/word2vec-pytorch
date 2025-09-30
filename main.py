import torch
import torch.nn as nn
from dataset import WikiText2, BasicEnglishTokenizer, build_vocab, collate_cbow
from torch.optim import Adam, SGD 
from model import CBOW
from functools import partial

import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader

ds = WikiText2() 
tokenizer = BasicEnglishTokenizer()
vocabulary = build_vocab(ds, tokenizer, min_freq=50) 

text_pipeline = lambda token : vocabulary(tokenizer(token)) 

dl = DataLoader(
  ds, 
  batch_size=256,
  shuffle=False,
  collate_fn = partial(collate_cbow, text_pipeline=text_pipeline)
)


model = CBOW(embedding_dim=300, vocab_size=len(vocabulary)) 

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

loss_curve = []

for epoch in range(1, 10+1): 
  total_loss = 0
  for i, (surrounding_tokens, target_token) in enumerate(dl): 
    print(f"Epoch {epoch} : Batch {i} / {len(dl)}")
    pred_token = model.forward(surrounding_tokens) 
    loss = criterion(pred_token, target_token)
    loss.backward() 
    optimizer.step() 

    total_loss += loss.detach().item() 

  loss_curve.append(total_loss)

torch.save(model, "saved/bruh.pth")


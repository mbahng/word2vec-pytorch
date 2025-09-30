import torch 
from dataset import WikiText2, BasicEnglishTokenizer, build_vocab, collate_cbow
from model import *
from functools import partial

from torch.utils.data import DataLoader

ds = WikiText2() 
tokenizer = BasicEnglishTokenizer()
vocabulary = build_vocab(ds, tokenizer, min_freq=50) 

text_pipeline = lambda token : vocabulary(tokenizer(token)) 

dl = DataLoader(
  ds, 
  batch_size=32,
  shuffle=False,
  collate_fn = partial(collate_cbow, text_pipeline=text_pipeline)
)

for batch in dl: 
  print(batch)



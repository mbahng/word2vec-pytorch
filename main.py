import torch 
from torch.utils.data import DataLoader 
from dataset import WikiText2
from model import *

def get_dataloader(): 

  ds = WikiText2()  
  dl = DataLoader(
    ds
  )
  return ds

if __name__ == "__main__": 
  ds = get_dataloader() 
  print(len(ds))

from torch.utils.data import Dataset
import kagglehub
import os

class WikiText2(Dataset): 

  def __init__(self): 
    self.ds_path = os.path.join(kagglehub.dataset_download("rohitgr/wikitext"), "wikitext-2") 
    self.data = []
    with open(os.path.join(self.ds_path, "wiki.train.tokens")) as f: 
      self.data = [stripped_line for line in f.readlines() if (stripped_line := line.strip())]
    
  def __getitem__(self, idx): 
    return self.data[idx]

  def __len__(self): 
    return len(self.data)


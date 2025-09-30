from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
from collections import Counter
from .tokenizer import Tokenizer

class Vocab: 

  def __init__(self, vocab: list): 
    self.vocab = vocab
    self.stoi = self._get_stoi() 
    self.itos = self._get_itos()

  def __len__(self): 
    return len(self.vocab)

  def __contains__(self, token: str) -> bool: 
    return token in self.vocab 

  def _get_stoi(self) -> dict[str, int]: 
    return { token : i for i, token in enumerate(self.vocab)}
  
  def _get_itos(self) -> dict[str, int]: 
    return { i : token for i, token in enumerate(self.vocab)}  # type: ignore

  def __call__(self, tokens: list[str]) -> list[int]: 
    return [self.stoi.get(t, 0) for t in tokens]

def build_vocab(dataset: Dataset, tokenizer: Tokenizer, min_freq: int = 100): 
  """Make a map of indices to tokens for each sentence in dataset""" 

  counter = Counter()
  for tokens in map(tokenizer, dataset.data):  # type: ignore
    counter.update(tokens)

  # First sort by descending frequency, then lexicographically
  sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  ordered_dict = OrderedDict(sorted_by_freq_tuples)

  ordered_dict.pop("<unk>", None)

  tokens = []
  # Save room for special tokens
  for token, freq in ordered_dict.items():
      if freq >= min_freq:
          tokens.append(token)

  tokens[0:0] = ["<unk>"]

  return Vocab(tokens)


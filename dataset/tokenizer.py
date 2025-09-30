from abc import ABC, abstractmethod
import re


_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]

_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]

_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

class Tokenizer(ABC): 

  @abstractmethod 
  def __call__(self, sentence: str): 
    ...

class SplitTokenizer(Tokenizer): 

  def __call__(self, sentence: str): 
    return sentence.split() 

class BasicEnglishTokenizer(Tokenizer): 

  def __call__(self, sentence: str): 
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """

    line = sentence.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


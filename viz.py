import torch 
from dataset import WikiText2, BasicEnglishTokenizer, build_vocab, collate_cbow
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ds = WikiText2() 
tokenizer = BasicEnglishTokenizer()
vocabulary = build_vocab(ds, tokenizer, min_freq=50) 

model = torch.load("saved/bruh.pth", weights_only=False) 
embeddings = model.embedding.weight.detach().clone().numpy()

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)  

words = ["boy", "girl", "king", "queen", "mother", "father"]

for word in words:
    idx = vocabulary.stoi[word]
    x, y = reduced[idx]
    plt.scatter(x, y)
    plt.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Word Embeddings Visualization')
plt.show()


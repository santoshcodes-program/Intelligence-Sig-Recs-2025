# word2vec_cbow.py

from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pandas as pd
import nltk
import multiprocessing

nltk.download('punkt')
nltk.download('punkt_tab')


dataset = load_dataset("lucadiliello/newsqa", split="train")

dataset = dataset.select(range(1000))


texts = [item['context'] for item in dataset if item.get('context')]



sentences = []
for text in texts:
    for sent in sent_tokenize(text):
        tokens = simple_preprocess(sent) 
        if len(tokens) > 2:
            sentences.append(tokens)




print("⏳ Training CBOW Word2Vec model...")

cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,            
    window=10,                 
    min_count=2,                
    sg=0,                       # CBOW
    workers=multiprocessing.cpu_count(),
    epochs=10
)



words = list(cbow_model.wv.key_to_index.keys())
vectors = [cbow_model.wv[word].tolist() for word in words]
df = pd.DataFrame({'word': words, 'embedding': vectors})
df.to_csv("cbow_embeddings.csv", index=False)


try:
    print("\n Most similar to 'india':")
    print(cbow_model.wv.most_similar("india", topn=5))
except KeyError:
    print(" 'india' not in vocabulary")

try:
    print("\n Word that doesn't match: ['king', 'queen', 'man', 'apple']:")
    print(cbow_model.wv.doesnt_match(['king', 'queen', 'man', 'apple']))
except KeyError:
    print("⚠️ Some words not in vocabulary")

try:
    print("\n Similarity between 'india' and 'pakistan':")
    print(cbow_model.wv.similarity("india", "pakistan"))
except KeyError:
    print(" 'india' or 'pakistan' not in vocabulary")



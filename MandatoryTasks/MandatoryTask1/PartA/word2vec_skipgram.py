from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')



dataset = load_dataset("lucadiliello/newsqa", split="train")
dataset = dataset.select(range(500))


texts = [item['context'] for item in dataset if item.get('context')]


sentences = []
for text in texts:
    for sent in sent_tokenize(text):
        tokens = simple_preprocess(sent)
        if len(tokens) > 2:
            sentences.append(tokens)


model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=2,
    sg=1,  # Skip-Gram
    workers=2
)


model.train(sentences, total_examples=model.corpus_count, epochs=10)


words = list(model.wv.key_to_index.keys())
vectors = [model.wv[word].tolist() for word in words]
df = pd.DataFrame({'word': words, 'embedding': vectors})
df.to_csv("word_embeddings.csv", index=False)




try:
    print("\n Similar words to 'india':")
    print(model.wv.most_similar("india", topn=5))
except KeyError:
    print("= 'india' not in vocabulary")

try:
    print("\n Word that doesn't match: ['king', 'queen', 'man', 'apple']")
    print(model.wv.doesnt_match(['king', 'queen', 'man', 'apple']))
except KeyError:
    print("Some of these words are not in the vocabulary")

try:
    print("\n Similarity between 'india' and 'pakistan':")
    print(model.wv.similarity("india", "pakistan"))
except KeyError:
    print(" 'india' or 'pakistan' not in vocabulary")

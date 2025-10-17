
"""
Using Hugging Face datasets library, we load the NewsQA dataset 
But since we dont need the entire thing, we are only taking  the first 1000 samples,
From each data entry, we extract the "context" field ,
here, we are  collecting a bunch of real world english sentences from news articles"""


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

#now, Word2Vec can not directly work on long paragraphs, it needs to be clean, tokenized sentences for this we will do text preprocessing
#For each sentence, we use Gensim simple_preprocess() to lowercase, clean, and tokenize the sentence and we are ignoring the sentences less than 2 words!
sentences = []
for text in texts:
    for sent in sent_tokenize(text):
        tokens = simple_preprocess(sent) 
        if len(tokens) > 2:
            sentences.append(tokens)





#we train a Word2Vec model using the CBOW (Continuous bag of words) approach
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,            
    window=10,                 
    min_count=2,                
    sg=0,                       # CBOW
    workers=multiprocessing.cpu_count(),
    epochs=10
)


"""Once the model has learned word meanings, we can extract the vocabulary and their corresponding 100-D vectors.
We store these in a dataframe and save them into a  csv file"""
words = list(cbow_model.wv.key_to_index.keys())
vectors = [cbow_model.wv[word].tolist() for word in words]
df = pd.DataFrame({'word': words, 'embedding': vectors})
df.to_csv("cbow_embeddings.csv", index=False)

# Here i tried doing some experiments do the model we trained 
try:
    print("Most similar to 'india':")
    print(cbow_model.wv.most_similar("india", topn=5))
except KeyError:
    print("'india' not in vocabulary")

try:
    print("Word that doesn't match: ['king', 'queen', 'man', 'apple']:")
    print(cbow_model.wv.doesnt_match(['king', 'queen', 'man', 'apple']))
except KeyError:
    print("Some words not in vocabulary")

try:
    print("Similarity between 'india' and 'pakistan':")
    print(cbow_model.wv.similarity("india", "pakistan"))
except KeyError:
    print("'india' or 'pakistan' not in vocabulary")



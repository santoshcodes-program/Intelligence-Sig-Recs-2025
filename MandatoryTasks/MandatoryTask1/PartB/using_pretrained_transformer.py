"""First, i took a pre-trained model called Helsinki-NLP/opus-mt-en-fr.
i loaded its tokenizer, which is like a tiny assistant that breaks sentences into words and numbers the model can understand"""
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import torch, random

model_name = "Helsinki-NLP/opus-mt-en-fr"


tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


dataset = load_dataset("ag_news", split="train")

#Then, I randomly picked 5 sentences from this dataset
def get_random_sentences(n=5):
    return [random.choice(dataset)["text"] for _ in range(n)]
#Before the translator can work, the sentences need to be converted into tokens,which the model can understand!
#the model generates translations from English tokens to French tokens.
#The model does not  output readable text at first it just gives numbers and special tokens
# therefore We use the tokenizer again to decode the tokens back into normal French sentences
def translate(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translations


sentences = get_random_sentences(5)
print("\n Random English Sentences:")
for s in sentences:
    print("-", s)

translated_sentences = translate(sentences)

print("\n Translations (English → French):")
for eng, fr in zip(sentences, translated_sentences):
    print(f"\nEN: {eng}\nFR: {fr}")


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from bnltk.tokenize import Tokenizers
from bnltk.stemmer import BanglaStemmer
from bnltk.tokenize import Tokenizers

from bnlp import corpus
from bnlp.corpus.util import remove_stopwords

from bnltk.stemmer import BanglaStemmer
bn_stemmer = BanglaStemmer()
tokenizer = Tokenizers()

text_final = []

for i in range(len(df["sentence"])):
    
    punc_less = re.sub('[^\u0980-\u09FF]',' ',str(df.sentence[i])) #removing unnecessary punctuation

    result = remove_stopwords(punc_less, stopwords)
    
    sentence_final_words = [bn_stemmer.stem(word) for word in result] 
    
    final_txt = ' '.join(sentence_final_words)
    
    text_final.append(final_txt)


def prediction(text):
    with open('vectorizer','rb') as file:
     vectoriser = pickle.load(file)
     text=vectoriser.transform(text)

    with open('sentiment_detection_model','rb') as file:
     model = pickle.load(file)
    # X = vectorizer.transform([text]).toarray()
    prediction=model.predict(model)
    prediction=prediction[0]

    sentiments={0:"Offensive",1:"Non-Offensive"}

    for i in sentiments.keys():
        if i == prediction:
            return sentiments[i]


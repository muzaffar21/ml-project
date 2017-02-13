from nltk import word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


class VectorizerTrainer:
    def __init__(self):
        pass
    # todo : add stemming and other features.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english',
                                 lowercase=True, token_pattern='[a-zA-Z0-9]+', strip_accents='unicode',
                                 tokenizer=word_tokenize)

    def generate_vector_matrix(self, corpus, vector_file):
        vec_matrix = self.vectorizer.fit_transform(corpus)
        print "vectors generated successfully , going to save vectorizer"
        pickle.dump(self.vectorizer, open(vector_file, "wb"))
        print "vectors saved successfully ,at " + vector_file

        return vec_matrix


def load_vectorizer(path):
    vectorizer = pickle.load(open(path, "rb"))
    print "vectorizer loaded successfully , ready to vectorize text"
    return vectorizer


def vectorize(test_data, vectorizer):
    vector = vectorizer.transform(test_data)
    return vector


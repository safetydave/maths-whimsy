from gensim.models import KeyedVectors
import numpy as np
import numpy.ma as ma
from numpy.linalg import norm
import tensorflow as tf
import tensorflow_hub as hub


class SimilarityModelW2V:

    def __init__(self, base_vocabulary, semantics=None):
        self.description = 'w2v'
        self.semantics = semantics
        if self.semantics is None:
            model = "data/GoogleNews-vectors-negative300.bin"
            self.semantics = KeyedVectors.load_word2vec_format(model, binary=True)
        self.vocabulary = [w for w in list(base_vocabulary) if w in self.semantics]
        self.embeddings = np.array([self.semantics[w] for w in self.vocabulary])

    def word_string(self, index):
        return self.vocabulary[index]

    def word_index(self, string):
        return self.vocabulary.index(string)

    def word_embedding(self, index):
        return self.embeddings[index]

    def random_guesses(self, n):
        return np.random.randint(len(self.vocabulary), size=n)

    def similarities(self, embedding):
        return self.semantics.cosine_similarities(embedding, self.embeddings)


def np_cosine(e, es):
    return np.dot(es, e) / (norm(es, axis=1) * norm(e))


class SimilarityModelUSE:

    def __init__(self, base_vocabulary):
        self.description = 'USE'
        use_ref = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.semantics = hub.load(use_ref)
        self.vocabulary = list(base_vocabulary) #[w for w in list(base_vocabulary) if w in self.semantics]
        self.embeddings = self.semantics(self.vocabulary).numpy()

    def word_string(self, index):
        return self.vocabulary[index]

    def word_index(self, string):
        return self.vocabulary.index(string)

    def word_embedding(self, index):
        return self.embeddings[index]

    def random_guesses(self, n):
        return np.random.randint(len(self.vocabulary), size=n)

    def similarities(self, embedding):
        return np_cosine(embedding, self.embeddings)

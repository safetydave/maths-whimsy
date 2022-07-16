import numpy as np
import numpy.ma as ma

class SimilarityModel:

    def __init__(self, semantics, base_vocabulary, precision=0.1, recall=1):
        self.semantics = semantics
        self.vocabulary = [w for w in list(base_vocabulary) if w in semantics]
        self.embeddings = np.array([self.semantics[w] for w in self.vocabulary])
        self.precision = precision
        self.recall = recall
        
    def word_string(self, index):
        return self.vocabulary[index]

    def word_index(self, string):
        return self.vocabulary.index(string)

    def word_embedding(self, index):
        return self.embeddings[index]
      
    def random_guesses(self, n):
        return np.random.randint(len(self.vocabulary), size=n)
   
    def similarity_rank(self, embedding):
        similarities = self.semantics.cosine_similarities(embedding, self.embeddings)
        return np.flip(np.argsort(similarities))        
  
    def cohort(self, index, similarity, mask):
        embedding = self.semantics[self.word_string(index)]
        similarities = self.semantics.cosine_similarities(embedding, self.embeddings)
        candidates = np.abs(similarities - similarity) < self.precision
        masked_candidates = ma.array(candidates, mask=mask)
        pool = masked_candidates.nonzero()[0]
        return np.random.choice(pool, size=int(round(self.recall * len(pool))))
      

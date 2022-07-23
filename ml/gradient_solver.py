import numpy as np


class GradientNetwork:

    def __init__(self, similarity):
        self.similarity = similarity
        self.nodes = {}
        self.spokes = {}
        self.top_score = -100
        self.top_index = -1
        self.top_delta = -1
        
    def d_embedding(self, i, j):
        return self.similarity.embeddings[j] - self.similarity.embeddings[i]

    def d_target_similarity(self, i, j):
        return self.nodes[j] - self.nodes[i]
          
    def add_spokes(self, i):
        new_spokes = []
        for j in self.spokes.keys():
            direction = self.d_embedding(i, j)
            gradient = self.d_target_similarity(i, j)
            new_spokes.append((direction, gradient, j))
            self.spokes[j].append((-direction, -gradient, i))
        self.spokes[i] = new_spokes       

    def add_node(self, i, score):
        self.nodes[i] = score
        if score > self.top_score:
            self.top_score = score
            self.top_index = i
        self.add_spokes(i)        

    def top_spoke(self, i):
        sorted_spokes = [s for s in sorted(self.spokes[i], key=lambda item: item[1])]
        return sorted_spokes[-1]

    def find_new_node(self, embedding):
        similarity_rank = np.flip(np.argsort(self.similarity.similarities(embedding)))
        j = 0
        while similarity_rank[j] in self.nodes:
            j = j + 1
        return similarity_rank[j]     

            
class GradientSolver:
  
    def __init__(self, similarity, seeds=3, log=False):
        self.similarity = similarity
        self.network = GradientNetwork(similarity)
        self.seeds = seeds
        self.log = log
        if self.log: self.logging = {}

    def log_append(self, label, value):
        if label not in self.logging:
            self.logging[label] = []
        self.logging[label].append(value)    
        
    def seed_guess(self):
        guess = self.similarity.random_guesses(1)[0]
        return guess

    def semantic_leap(self, node_index, spoke, distance):
        return self.similarity.word_embedding(node_index) + spoke[0] * distance

    def directed_guess(self):
        node_basis = self.network.top_index
        spoke_basis = self.network.top_spoke(node_basis)
        distance = np.random.rand(1) * 2
        embedding = self.semantic_leap(node_basis, spoke_basis, distance)
        guess = self.network.find_new_node(embedding)
        return guess, (node_basis, spoke_basis)
      
    def make_guess(self):
        guess = None
        basis = (-1, (None, 0, -1))
        if len(self.network.nodes) < self.seeds or len(self.network.nodes) < 2:
            guess = self.seed_guess()
        else:
            guess, basis = self.directed_guess()
        if self.log: self.log_append('basis', basis)
        return guess

    def merge_guess(self, guess, score, score_scaling):
        if self.log: self.log_append('guess', guess)
        self.network.add_node(guess, score)

    def guess_merged(self, guess):
        return guess in self.network.nodes
        
    def solve(self, semantle, max_guesses=500):
        if semantle.target not in self.similarity.vocabulary:
            return None, 0      
        eps = 0.1
        guess = self.make_guess()
        score = semantle.score_guess(self.similarity.word_string(guess))
        while score < semantle.score_scaling - eps:
            self.merge_guess(guess, score, semantle.score_scaling)
            guess = self.make_guess()
            score = semantle.score_guess(self.similarity.word_string(guess))
            if len(self.network.nodes) >= max_guesses:
                break                 
        if self.log: self.log_append('guess', guess)
        return guess, len(self.network.nodes) + 1
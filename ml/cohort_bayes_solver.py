import numpy as np
import numpy.ma as ma


class CohortBayesSolver:

    def __init__(self, similarity, log=False):
        self.similarity = similarity
        self.count = len(similarity.vocabulary)
        self.strength = ma.array(np.ones(self.count, dtype=int), mask=False)
        self.steps = 0
        self.log = log
        if self.log: self.logging = {}

    def log_append(self, label, value):
        if label not in self.logging:
            self.logging[label] = []
        self.logging[label].append(value)
            
    def index_t(self, t):
        sort_ids = self.strength.argsort(fill_value=-1)
        masked_count = ma.count_masked(self.strength)
        unmasked_range = self.count - masked_count
        position = masked_count + int(t * unmasked_range) - 1
        return sort_ids[position]    
        
    def make_guess(self):
        guess = self.index_t(1)
        self.steps = self.steps + 1
        if self.log: self.log_append('guess', guess)
        return guess
      
    def merge_guess(self, guess, score, score_scaling):
        self.strength[guess] = ma.masked
        norm_score = score / score_scaling
        cohort = self.similarity.cohort(guess, norm_score, ma.getmask(self.strength))
        if self.log: self.log_append('cohort_size', len(cohort))
        observations = np.zeros(self.count, dtype=int)
        observations[cohort] = max(0, int(round(score)))
        # strength when normalised approximates bayesian probabilities
        self.strength = self.strength + observations
        
    def solve(self, semantle):
        if semantle.target not in self.similarity.vocabulary:
            return None, 0
        eps = 0.1
        guess = self.make_guess()
        score = semantle.score_guess(self.similarity.word_string(guess))
        while score < semantle.score_scaling - eps:
            self.merge_guess(guess, score, semantle.score_scaling)
            guess = self.make_guess()
            score = semantle.score_guess(self.similarity.word_string(guess))
            if self.log: self.log_extended(semantle)
        if self.log: self.log_append('cohort_size', 0)
        return guess, self.steps
      
    def log_extended(self, semantle):
        target = self.similarity.word_index(semantle.target)
        self.log_append('target_strength', self.strength[target])
        self.log_append('max_strength', np.max(self.strength))
        self.log_append('strength_history', ma.getdata(self.strength))
        sort_ids = self.strength.argsort(fill_value=-1)
        target_id = np.argwhere(sort_ids == target).flatten()[0]
        self.log_append('target_rank', self.count - target_id)
      
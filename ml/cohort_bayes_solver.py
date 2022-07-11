import numpy as np
import numpy.ma as ma


class CohortBayesSolver:

    def __init__(self, similarity, log=False):
        self.similarity = similarity
        self.count = len(similarity.vocabulary)
        self.strength = ma.array(np.ones(self.count, dtype=int), mask=False)
        self.steps = 0
        if log:
            self.logging = {'guess': [],
                            'target_strength': [],
                            'max_strength': [],
                            'target_rank': [],
                            'strength': []}

    def index_t(self, t):
        sort_ids = self.strength.argsort(fill_value=-1)
        masked_count = ma.count_masked(self.strength)
        unmasked_range = self.count - masked_count
        position = masked_count + int(t * unmasked_range) - 1
        return sort_ids[position]    
        
    def make_guess(self, t):
        return self.index_t(t)

    def merge_guess(self, guess, score, score_scaling):
        self.strength[guess] = ma.masked
        norm_score = score / score_scaling
        cohort = self.similarity.cohort(guess, norm_score, ma.getmask(self.strength))
        observations = np.zeros(self.count, dtype=int)
        observations[cohort] = max(0, int(round(score)))
        # strength when normalised approximates bayesian probabilities
        self.strength = self.strength + observations
        
    def solve(self, semantle):
        if semantle.target not in self.similarity.vocabulary:
            return None, 0
        eps = 0.1
        guess = self.make_guess(1)
        score = semantle.score_guess(self.similarity.word_string(guess))
        while score < semantle.score_scaling - eps:
            self.merge_guess(guess, score, semantle.score_scaling)
            guess = self.make_guess(1)
            score = semantle.score_guess(self.similarity.word_string(guess))
            self.steps = self.steps + 1
            if self.logging is not None:
              self.do_log(guess, semantle)
        return guess, self.steps
      
    def do_log(self, guess, semantle):
        self.logging['guess'].append(guess)
        target = self.similarity.word_index(semantle.target)
        self.logging['target_strength'].append(self.strength[target])
        self.logging['max_strength'].append(np.max(self.strength))
        self.logging['strength'].append(ma.getdata(self.strength))
        sort_ids = self.strength.argsort(fill_value=-1)
        target_id = np.argwhere(sort_ids == target).flatten()[0]
        self.logging['target_rank'].append(self.count - target_id)
      